# Base image for the CONTAINED langchain-rine dev/test rig.
# The LangChain tree (langchain, langchain-core, langgraph, langchain-openai,
# langchain-tests, ...) installs INSIDE the container only (into a venv on a named volume;
# see ../compose.langchain.yml). NOTHING langchain-related is ever installed on the host —
# same rule as the crewai / OpenClaw rigs. The local rine-sdk and rine-langchain are
# bind-mounted and installed editable, so the rig consumes UNRELEASED SDK fixes (Phase-0).
FROM python:3.12-slim

# git for any VCS deps; build-essential for native wheels pulled by the LangChain tree.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv: fast resolver/installer; populates the named-volume venv on first run.
RUN pip install --no-cache-dir uv

# The venv lives on a NAMED VOLUME (mounted at /opt/venv in compose) so heavy deps are
# cached across runs and never land on the host tree. Activate it for every command.
ENV VENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CI=true

WORKDIR /work/rine-langchain

# Idempotent bootstrap: ensure the named-volume venv exists and the bind-mounted
# rine-sdk + langchain-rine[dev] are installed editable into it, then exec the command.
# `langchain-rine[dev]` pulls langchain-core/langgraph/langchain-openai/langchain-tests;
# `rine` resolves to the bind-mounted ./rine-sdk (editable) — overriding the published
# floor for dev/tests.
COPY entrypoint.sh /usr/local/bin/rig-entrypoint
RUN chmod +x /usr/local/bin/rig-entrypoint

ENTRYPOINT ["rig-entrypoint"]
CMD ["python", "-c", "import importlib.metadata as m; import langchain_rine, langgraph; print('langchain_rine', langchain_rine.__version__, '/ langgraph', m.version('langgraph'))"]
