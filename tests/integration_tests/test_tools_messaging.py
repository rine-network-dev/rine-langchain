"""``langchain-tests`` conformance for the five messaging tools.

One ``ToolsUnitTests`` + one ``ToolsIntegrationTests`` subclass per tool. The integration
suite invokes each tool over the real SDK against respx-mocked endpoints; the happy paths
mirror ``tests/unit_tests`` (send confirm, send-and-wait timeout, empty inbox, MLS read,
reply id). Every tool returns a ``str``, so the suite's artifact assertion never fires.
"""

from __future__ import annotations

import httpx
import respx
from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests
from rine._constants import EncryptionVersion

from langchain_rine import (
    RineInboxTool,
    RineReadTool,
    RineReplyTool,
    RineSendAndWaitTool,
    RineSendTool,
)
from tests.conftest import (
    AGENT_ID,
    API_URL,
    CONVERSATION_ID,
    MESSAGE_ID,
    OTHER_AGENT_ID,
    OTHER_HANDLE,
    REPLY_ID,
    message_payload,
    mock_agents,
    mock_recipient_keys,
    mock_token,
    mock_webfinger,
    sent_message_payload,
)
from tests.integration_tests._conformance import RineToolConformance


class _SendSpec(RineToolConformance):
    tool_cls = RineSendTool
    invoke_args = {"to": OTHER_HANDLE, "body": "hi"}
    needs_other_keys = True

    def install_mocks(self) -> None:
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(self._cfg)
        respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(200, json=sent_message_payload())
        )


class TestRineSendUnit(_SendSpec, ToolsUnitTests):
    pass


class TestRineSendIntegration(_SendSpec, ToolsIntegrationTests):
    pass


class _SendAndWaitSpec(RineToolConformance):
    tool_cls = RineSendAndWaitTool
    invoke_args = {"to": OTHER_HANDLE, "body": "hi", "wait_seconds": 5}
    needs_other_keys = True

    def install_mocks(self) -> None:
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(self._cfg)
        # No reply within the window → the deterministic timeout-string path.
        respx.post(f"{API_URL}/messages/sync").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": sent_message_payload(),
                    "reply": None,
                    "conversation_id": CONVERSATION_ID,
                    "status": "timeout",
                },
            )
        )


class TestRineSendAndWaitUnit(_SendAndWaitSpec, ToolsUnitTests):
    pass


class TestRineSendAndWaitIntegration(_SendAndWaitSpec, ToolsIntegrationTests):
    pass


class _InboxSpec(RineToolConformance):
    tool_cls = RineInboxTool
    invoke_args: dict[str, object] = {}

    def install_mocks(self) -> None:
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )
        respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 0})
        )


class TestRineInboxUnit(_InboxSpec, ToolsUnitTests):
    pass


class TestRineInboxIntegration(_InboxSpec, ToolsIntegrationTests):
    pass


class _ReadSpec(RineToolConformance):
    tool_cls = RineReadTool
    invoke_args = {"message_id": MESSAGE_ID}

    def install_mocks(self) -> None:
        mock_token()
        mock_agents()
        # An MLS message renders its decrypt_error string — no key fetch needed.
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(
                200,
                json=message_payload(
                    encryption_version=EncryptionVersion.MLS_V1,
                    encrypted_payload="Q0lQSEVSVEVYVA==",
                ),
            )
        )


class TestRineReadUnit(_ReadSpec, ToolsUnitTests):
    pass


class TestRineReadIntegration(_ReadSpec, ToolsIntegrationTests):
    pass


class _ReplySpec(RineToolConformance):
    tool_cls = RineReplyTool
    invoke_args = {"message_id": MESSAGE_ID, "body": "hi"}
    needs_other_keys = True

    def install_mocks(self) -> None:
        mock_token()
        mock_agents()
        # reply() reads the original (to find the sender), then encrypts to them.
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(
                200,
                json=message_payload(
                    encryption_version=EncryptionVersion.HPKE_V1,
                    encrypted_payload="ZHVtbXk=",
                    from_agent_id=OTHER_AGENT_ID,
                    to_agent_id=AGENT_ID,
                ),
            )
        )
        mock_recipient_keys(self._cfg)
        respx.post(f"{API_URL}/messages/{MESSAGE_ID}/reply").mock(
            return_value=httpx.Response(200, json={**sent_message_payload(), "id": REPLY_ID})
        )


class TestRineReplyUnit(_ReplySpec, ToolsUnitTests):
    pass


class TestRineReplyIntegration(_ReplySpec, ToolsIntegrationTests):
    pass
