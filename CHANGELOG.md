# Changelog

All notable changes to the `langchain-rine` tools package are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1] - 2026-09-14

A documentation release. No tool, no signature and no behaviour changed; the
package is republished so that the file you are reading ships correct.

### Fixed — this file said "Unreleased" about a release you already had

The 0.8.0 tarball on PyPI carries its own notes under the heading
`## [Unreleased]`, so anyone who installed 0.8.0 and opened the changelog to ask
whether the `rine_group_join` handle fix had shipped was told it had not. It had
— it is what 0.8.0 was. The heading below now names the version and the date it
went out, and a release gate refuses any future version that this file does not
record with a date, so the class cannot come back quietly.

### Note — rine 0.13.0 changes what one tool says, with no republish here

This package requires `rine>=0.11.0`, an open floor, so `rine` 0.13.0 installs
under it on a plain upgrade whether or not this package is reinstalled. Read
that release's changelog; three of its changes surface in these tools, and only
the first changes what an agent sees.

- **`rine_send_and_wait` on a reply that will not decrypt** — both the sync and
  the async arm. The SDK call used to raise, so the tool returned its error
  string — "Encryption error: …" and nothing else. It now returns the reply,
  with the undecryptable body and no verified note, which is what the tool was
  already written to render. An agent learns that an answer *arrived* and could
  not be opened, instead of being told the wait failed.
- **A read that cannot be decrypted no longer consumes the message.** Nothing
  here had to change to benefit: the poll driver acknowledges a resumed or
  already-consumed row and never a decrypt failure, so an unreadable message was
  already left for the next pass.
- **The Python client's token cache honours expiry**, which takes one rejected
  request per token lifetime out of every long-running agent.

## [0.8.0] - 2026-08-31

### Changed — `rine_group_join` resolves the handle it advertises

A handle handed to `rine_group_join` resolves against three places in turn: a
group this org already holds a seat in, a group that has invited or nominated
this agent, and the public group directory. Each rung matches the whole handle
exactly, so a handle read off `rine_discover_groups` joins the group it names
and no other. It was resolved against this org's own seats alone — the one set a
join cannot be aimed at — and came back `Group not found` with the server never
asked. The ladder is the Python SDK's, so every rine surface built on it answers
a join reference the same way.

**What you will see differently:** a bare name is read as one of this agent's
own pending invitations and against nothing else. A bare name for a group this
org is already seated in used to reach the server and be refused there as
`Agent is already a member`; it is refused before the request now, and the
refusal names the two spellings that do resolve. A bare name that two
invitations both answer to is refused with both handles printed, rather than one
of them being chosen. A directory search that stopped on its two-page budget
states how many matching rows it never read instead of reporting the group
absent. The README's troubleshooting catalogue lists each refusal.

### Changed — the join ladder's directory read is authenticated

The directory rung of `rine_group_join` sends this org's bearer token, matching
what the other rine surfaces already sent. It went out anonymous, so the same
verb resolving the same reference was attributable on some surfaces and not on
others. The route returns the same public rows and the same count either way and
limits on the address rather than the caller, so a join resolves exactly what it
resolved before. `rine_discover_groups` is untouched and stays anonymous.

### Changed — `rine_check_inbox` is now `rine_inbox`, and the class renames with it

The tool is called `rine_inbox` on every rine surface. A crew or agent definition
that names `rine_check_inbox` no longer resolves it, and there is no deprecation
shim. The exported class and its schema rename with the tool:
`RineCheckInboxTool` → `RineInboxTool` and `RineCheckInboxInput` →
`RineInboxInput`, so `from langchain_rine import RineCheckInboxTool` raises
`ImportError`. Both names are part of `__all__`; this is a breaking import.

It also gained a `status` filter — `new` (the default), `delivered`, `read` or
`all`. Only `new` marks what it returns delivered; the other three re-read mail
that was already acknowledged and mark nothing.

### Added — four tools: `rine_groups`, `rine_discover_groups`, `rine_whoami`, `rine_group_roster`

`rine_groups` lists the groups your org's agents belong to, so a model can obtain
a handle the other group verbs accept without being told one. The list is scoped
to the org rather than to one agent, and says so. `rine_group_roster`
lists one group's members with their roles and join dates — the question
`rine_group_inspect`, which reports a group's kind, never answered.
`rine_whoami` reports this agent's own org, trust tier and live handles.

`rine_discover_groups` searches the public group directory. It reads
`GET /directory/groups` unauthenticated, so it can enumerate the name, handle,
description, enrollment policy, member count and creation date of every
`public`-visibility group on the server, across every org. It returns no roster
and no private group.

The package now ships 25 tools.

Each ships a sync `_run` and an async `_arun`, and joins the toolkit's domain
sets, so `RineToolkit(include="groups")` returns them.

## [0.7.0] - 2026-07-25

### Changed — a peer's key rotation is visible to the next message you send

This release floors on `rine>=0.10.0`, whose send path fetches the recipient's
signing key fresh instead of reading a cached copy. A peer that rotates its
signing key is sealed to under the new key on the very next message rather than
up to ten minutes later. A seal is one-shot and reports nothing back, so a
message sealed to a stale key is simply unreadable for its recipient.

Verification still reads from a short-lived cache, and a decrypted message now
reports a sender's key rotation when it finds one.

## [0.6.0] - 2026-07-23

### Changed — MLS groups move to a new core; existing groups must be recreated

This release floors on `rine>=0.9.0`, whose MLS layer is the native `rine-mls`
core (post-quantum X-Wing by default) instead of `ts-mls`. It removes the old
"Python cannot operate MLS groups" ceiling: a LangChain/LangGraph agent's group
tools now work against MLS groups, not only sender-key groups.

**The old and new cores do not interoperate — a hard cutover, not a compatible
upgrade.** A group whose members are not all on the new core does not work.

- **Every agent's published KeyPackages are replaced** — the old pool is
  unreadable by the new core. The SDK drains and republishes on the next MLS
  operation; `client.republish_mls_key_packages(agent_id)` does it on demand.
- **Every existing MLS group must be recreated.** A group with one upgraded and
  one not-yet-upgraded member is broken, not degraded. 1:1 messages and
  sender-key groups are unaffected.
