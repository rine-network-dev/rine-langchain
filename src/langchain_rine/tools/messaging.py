"""1:1 + group messaging tools (SPEC §6.1–6.5, §6.9).

Five ``BaseTool``s over the rine SDK: send, send-and-wait, check-inbox, read, reply.
Each ships both a sync ``_run`` (over ``SyncRineClient``) and an async ``_arun`` (over the
async ``RineClient``) — the two bodies are identical apart from ``await``ing the SDK call
and resolving the async client (``CursorPage`` stays sync-iterable via ``list(page)``,
PHASE0_CONTRACT §0). Group messaging is not a separate tool — a ``to`` starting with ``#``
routes through ``RineSendTool`` and group mail lands in the inbox/read path (SPEC §6.9).
Every body maps SDK errors to a string via ``_format_error`` — never a raw traceback,
never ciphertext. Argument schemas live in ``_schemas.py``.
"""

import logging

from pydantic import BaseModel
from rine import UnsupportedTargetError

from langchain_rine._client import (
    AsyncCallbackManagerForToolRun,
    BaseRineTool,
    CallbackManagerForToolRun,
    _format_error,
)
from langchain_rine._format import (
    _verified_note,
    render_inbox,
    render_message_body,
    render_single_message,
    render_thread,
)
from langchain_rine.tools._schemas import (
    RineCheckInboxInput,
    RineReadInput,
    RineReplyInput,
    RineSendAndWaitInput,
    RineSendInput,
    RineThreadInput,
)

logger = logging.getLogger("langchain_rine.messaging")


class RineSendTool(BaseRineTool):
    """Send an encrypted 1:1 or group message on the rine network."""

    name: str = "rine_send"
    description: str = (
        "Send an encrypted message to another agent (or a group) on the rine network. "
        "Performs a real, irreversible network send. Use to='handle@org' for 1:1 or "
        "to='#group@org' for a group."
    )
    args_schema: type[BaseModel] = RineSendInput

    def _run(
        self,
        to: str,
        body: str,
        message_type: str = "rine.v1.task_request",
        idempotency_key: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            msg = self._get_client().send(
                to, {"text": body}, message_type=message_type, idempotency_key=idempotency_key
            )
        except Exception as err:  # noqa: BLE001 — surface as a tool message, never crash
            return _format_error(err)
        return f"Sent message {msg.id} to {to} (conversation {msg.conversation_id})."

    async def _arun(
        self,
        to: str,
        body: str,
        message_type: str = "rine.v1.task_request",
        idempotency_key: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            msg = await self._aget_client().send(
                to, {"text": body}, message_type=message_type, idempotency_key=idempotency_key
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return f"Sent message {msg.id} to {to} (conversation {msg.conversation_id})."


class RineSendAndWaitTool(BaseRineTool):
    """Send a message and block until a reply arrives or the timeout elapses."""

    name: str = "rine_send_and_wait"
    description: str = (
        "Send a message to another agent and block until they reply or the timeout "
        "elapses. The delegate-and-await primitive. Performs a real network send. "
        "1:1 only (not groups)."
    )
    args_schema: type[BaseModel] = RineSendAndWaitInput

    def _run(
        self,
        to: str,
        body: str,
        wait_seconds: int = 30,
        message_type: str = "rine.v1.task_request",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            result = self._get_client().send_and_wait(
                to, {"text": body}, timeout=float(wait_seconds), message_type=message_type
            )
        except UnsupportedTargetError:
            return "send_and_wait is 1:1 only; use rine_send for groups."
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        if result.reply is not None:
            body_text = render_message_body(result.reply)
            if result.reply.decrypt_error is None:
                return f"Reply from {to}: {body_text} ({_verified_note(result.reply)})"
            return f"Reply from {to}: {body_text}"
        return f"Sent {result.sent.id}; no reply within {wait_seconds}s."

    async def _arun(
        self,
        to: str,
        body: str,
        wait_seconds: int = 30,
        message_type: str = "rine.v1.task_request",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            result = await self._aget_client().send_and_wait(
                to, {"text": body}, timeout=float(wait_seconds), message_type=message_type
            )
        except UnsupportedTargetError:
            return "send_and_wait is 1:1 only; use rine_send for groups."
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        if result.reply is not None:
            body_text = render_message_body(result.reply)
            if result.reply.decrypt_error is None:
                return f"Reply from {to}: {body_text} ({_verified_note(result.reply)})"
            return f"Reply from {to}: {body_text}"
        return f"Sent {result.sent.id}; no reply within {wait_seconds}s."


class RineCheckInboxTool(BaseRineTool):
    """Fetch NEW (undelivered) messages and mark them delivered."""

    name: str = "rine_check_inbox"
    description: str = (
        "Check the rine inbox for NEW (undelivered) messages and return their decrypted "
        "contents. Marks them delivered so the next check returns only newer mail."
    )
    args_schema: type[BaseModel] = RineCheckInboxInput

    def _run(self, limit: int = 20, run_manager: CallbackManagerForToolRun | None = None) -> str:
        ack_failed = False
        try:
            client = self._get_client()
            page = client.inbox(status="new", limit=limit, agent=self.agent)
            messages = list(page)
            deliverable = [str(m.id) for m in messages if m.decrypt_error is None]
            if deliverable:
                # mark_delivered resolves the same agent inbox() used (public SDK
                # surface §16). Best-effort: an ack failure must never lose the read —
                # but it IS surfaced (WARNING log + a note) so the LLM knows these
                # messages may reappear on the next poll (SPEC §6.3).
                try:
                    client.mark_delivered(deliverable, agent=self.agent)
                except Exception:  # noqa: BLE001 — ack is best-effort, keep the messages
                    ack_failed = True
                    logger.warning(
                        "rine_check_inbox: mark_delivered failed; %d message(s) may reappear",
                        len(deliverable),
                        exc_info=True,
                    )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        rendered = render_inbox(messages)
        if ack_failed:
            rendered += " (note: could not mark delivered; these may reappear)"
        return rendered

    async def _arun(
        self, limit: int = 20, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> str:
        ack_failed = False
        try:
            client = self._aget_client()
            page = await client.inbox(status="new", limit=limit, agent=self.agent)
            messages = list(page)  # CursorPage is sync-iterable (PHASE0_CONTRACT §0)
            deliverable = [str(m.id) for m in messages if m.decrypt_error is None]
            if deliverable:
                try:
                    await client.mark_delivered(deliverable, agent=self.agent)
                except Exception:  # noqa: BLE001 — ack is best-effort, keep the messages
                    ack_failed = True
                    logger.warning(
                        "rine_check_inbox: mark_delivered failed; %d message(s) may reappear",
                        len(deliverable),
                        exc_info=True,
                    )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        rendered = render_inbox(messages)
        if ack_failed:
            rendered += " (note: could not mark delivered; these may reappear)"
        return rendered


class RineReadTool(BaseRineTool):
    """Fetch and decrypt a single rine message by id."""

    name: str = "rine_read"
    description: str = "Fetch and decrypt a single rine message by id."
    args_schema: type[BaseModel] = RineReadInput

    def _run(self, message_id: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            msg = self._get_client().read(message_id, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_single_message(msg)

    async def _arun(
        self, message_id: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> str:
        try:
            msg = await self._aget_client().read(message_id, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_single_message(msg)


class RineReplyTool(BaseRineTool):
    """Reply in-thread to a rine message."""

    name: str = "rine_reply"
    description: str = (
        "Reply in-thread to a rine message. Performs a real, irreversible network send."
    )
    args_schema: type[BaseModel] = RineReplyInput

    def _run(
        self,
        message_id: str,
        body: str,
        message_type: str = "rine.v1.task_response",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            reply = self._get_client().reply(
                message_id, {"text": body}, message_type=message_type, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return f"Replied to {message_id} -> {reply.id}."

    async def _arun(
        self,
        message_id: str,
        body: str,
        message_type: str = "rine.v1.task_response",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            reply = await self._aget_client().reply(
                message_id, {"text": body}, message_type=message_type, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return f"Replied to {message_id} -> {reply.id}."


class RineThreadTool(BaseRineTool):
    """Fetch the both-sided, decrypted transcript of a conversation."""

    name: str = "rine_thread"
    description: str = (
        "Fetch the both-sided, decrypted transcript of a conversation by its UUID. "
        "Returns every turn (yours and the peer's) oldest-to-newest, role-tagged. "
        "A turn you cannot decrypt renders [unavailable]."
    )
    args_schema: type[BaseModel] = RineThreadInput

    def _run(
        self,
        conversation_id: str,
        limit: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            entries = self._get_client().thread(conversation_id, agent=self.agent, limit=limit)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_thread(entries)

    async def _arun(
        self,
        conversation_id: str,
        limit: int | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            entries = await self._aget_client().thread(
                conversation_id, agent=self.agent, limit=limit
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_thread(entries)


__all__ = [
    "RineSendTool",
    "RineSendAndWaitTool",
    "RineCheckInboxTool",
    "RineReadTool",
    "RineReplyTool",
    "RineThreadTool",
]
