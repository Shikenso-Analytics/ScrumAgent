# ScrumAgent v2.0 — Lokale Optimierung & Finalisierung

## Kontext

Branch `claude/scrumagent-langgraph-mcp-bz738` enthält eine vollständige Migration des ScrumAgent von Multi-Agent-Supervisor-Architektur zu einem Single-Agent mit MCP (Model Context Protocol). Die Migration wurde in einer eingeschränkten Cloud-Umgebung durchgeführt — ohne Docker, npm, uv, und mit kaputtem C-Compiler (sgmllib3k/cryptography/cffi konnten nicht bauen).

### Aktuelle Architektur (bereits implementiert)

- `scrumagent/agent.py` — Single ReAct Agent (`create_react_agent`) mit `MultiServerMCPClient`
- `mcp_servers/discord_chroma_server.py` — Custom MCP Server (ChromaDB semantic search)
- `config/mcp_config.yaml` — 4 MCP Server: Taiga, GitHub, Discord Chroma, Discord API
- `scrumagent/main_discord_bot.py` — Discord Bot nutzt `ScrumAgent` Klasse
- `pyproject.toml` — Ersetzt requirements.txt, LangChain 1.x / LangGraph 1.x

### Was in der Cloud NICHT möglich war

Die folgenden Punkte konnten remote nicht umgesetzt werden und brauchen lokale Arbeit.

---

## Aufgabe 1: uv Setup & Dependencies

Die Cloud hatte kein `uv`. Dependencies wurden mit `pip install` installiert.

- Initialisiere das Projekt mit `uv`: `uv sync` basierend auf der bestehenden `pyproject.toml`
- Stelle sicher, dass ALLE Dependencies sauber installieren, insbesondere:
  - `arxiv` (braucht `feedparser` → `sgmllib3k` — C-Extension, brauchte funktionierenden Compiler)
  - `wikipedia` (braucht ebenfalls `feedparser`)
  - `chromadb` (braucht `onnxruntime` oder `sqlite3`)
  - `langchain-taiga>=1.5.0` — Falls nicht auf PyPI: `uv pip install git+https://github.com/Shikenso-Analytics/langchain-taiga@main`
- Generiere ein `uv.lock` File
- Prüfe ob `_build_web_tools()` in `scrumagent/agent.py` dann alle 4 Tools erstellt (DuckDuckGo, ArXiv, YouTube, Wikipedia) — nicht nur 2 wie in der Cloud

## Aufgabe 2: MCP Server testen

Jeder MCP Server muss einzeln getestet werden:

### 2a) Taiga MCP

```bash
python -m langchain_taiga.mcp_server
```

- Prüfe ob es startet und Tools listet
- Braucht `TAIGA_API_URL`, `TAIGA_TOKEN` etc. in `.env`

### 2b) GitHub MCP

```bash
docker pull ghcr.io/github/github-mcp-server
docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server
```

- Braucht `GITHUB_PERSONAL_ACCESS_TOKEN`

### 2c) Discord Chroma MCP

```bash
python mcp_servers/discord_chroma_server.py
```

- Braucht `CHROMA_DB_PATH`, `OPENAI_API_KEY`

### 2d) Discord API MCP

```bash
npx -y mcp-discord
```

- npm-Paket. Braucht `DISCORD_TOKEN` als env var
- Prüfe ob das Paket `mcp-discord` von barryyip0625 ist (nicht ein anderes gleichnamiges)

## Aufgabe 3: Env-Propagation an MCP Subprozesse

**Kritischer Bug-Kandidat**: `MultiServerMCPClient` spawnt MCP Server als Subprozesse via stdio. Prüfe ob Environment-Variablen automatisch an die Subprozesse weitergegeben werden. Falls nicht, muss `config/mcp_config.yaml` um `env:`-Blöcke erweitert werden und `_load_mcp_config()` in `agent.py` muss diese parsen.

Betroffene Variablen:

- **Taiga**: `TAIGA_API_URL`, `TAIGA_URL`, `TAIGA_TOKEN` (oder `TAIGA_USERNAME`/`TAIGA_PASSWORD`)
- **GitHub Docker**: `-e GITHUB_PERSONAL_ACCESS_TOKEN` ist bereits im Docker-Kommando, aber der Host-Prozess muss die Variable haben
- **Discord Chroma**: `CHROMA_DB_PATH`, `CHROMA_DB_DISCORD_CHAT_DATA_NAME`, `OPENAI_API_KEY`
- **Discord API**: `DISCORD_TOKEN`

Schau in den Source Code von `langchain-mcp-adapters` (`MultiServerMCPClient`) ob `env` als Parameter unterstützt wird. Falls ja, nutze es in der Config. Falls nein, stelle sicher dass `load_dotenv()` vor dem MCP-Start aufgerufen wird und die Variablen im Prozess-Environment sind.

## Aufgabe 4: Sync/Async Invoke Fix

In `main_discord_bot.py` wird `scrum_agent.invoke()` (synchron) via `run_in_executor` aufgerufen. Aber `create_react_agent` erzeugt einen LangGraph, dessen `.invoke()` intern async Operations nutzen kann. Das kann zu Event-Loop-Konflikten führen.

Prüfe:

1. Ob `graph.invoke()` wirklich synchron funktioniert (LangGraph unterstützt beides)
2. Falls es Probleme gibt: Ersetze `run_in_executor` + sync `invoke` durch direktes `await scrum_agent.ainvoke()` in einer async Methode
3. Teste mit einem echten Discord-Bot-Lauf

## Aufgabe 5: Multi-LLM Support

`agent.py` nutzt hardcoded `ChatOpenAI`. Der Plan sah vor, dass auch Anthropic und Ollama unterstützt werden (env var `SCRUM_AGENT_MODEL`).

Implementiere eine Factory-Funktion:

```python
def _build_llm():
    model = os.getenv("SCRUM_AGENT_MODEL", "gpt-4o")
    temp = float(os.getenv("SCRUM_AGENT_TEMPERATURE", "0"))

    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temp)
    elif model.startswith("ollama/"):
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model.removeprefix("ollama/"), temperature=temp)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model_name=model, temperature=temp)
```

## Aufgabe 6: LangSmith Observability

LangSmith Tracing ist in `.env.example` schon vorbereitet (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`), aber es fehlen granulare Callback-Handler.

- Prüfe ob `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` automatisch alle Tool-Calls traced (LangGraph 1.x sollte das nativ tun)
- Falls nicht: Füge explizite Callbacks hinzu
- Teste einen Durchlauf und prüfe das LangSmith Dashboard

## Aufgabe 7: Integration Tests

Erstelle `tests/test_integration.py` mit echten MCP-Verbindungen (nur wenn `.env` konfiguriert ist):

```python
import pytest
import os

@pytest.mark.skipif(not os.getenv("TAIGA_TOKEN"), reason="No Taiga credentials")
@pytest.mark.asyncio
async def test_taiga_mcp_connection():
    """Start ScrumAgent, verify Taiga tools are available."""
    ...

@pytest.mark.skipif(not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"), reason="No GitHub token")
@pytest.mark.asyncio
async def test_github_mcp_connection():
    """Start ScrumAgent, verify GitHub tools are available."""
    ...
```

## Aufgabe 8: Security Review

- Prüfe dass keine Secrets in committed Files sind (`.env` ist in `.gitignore`)
- Prüfe dass der Discord Chroma MCP Server keine Injection-Anfälligkeiten hat (ChromaDB `where`-Filter in `discord_channel_history`)
- Prüfe dass der GitHub MCP Docker-Container keine übermäßigen Rechte bekommt
- Stelle sicher dass `GITHUB_PERSONAL_ACCESS_TOKEN` minimal-scope hat (nur `repo:read`, `issues:read`)

## Aufgabe 9: Cleanup & Polish

- Entferne die `try/except ImportError` Workarounds in `_build_web_tools()` falls alle Dependencies jetzt sauber installieren — oder behalte sie als defensive Programmierung (deine Entscheidung)
- Prüfe ob `langgraph.json` korrekt ist für LangGraph Cloud Deployment (`ScrumAgent` ist eine Klasse, kein callable — möglicherweise braucht es einen Wrapper)
- Teste `langgraph dev` oder `langgraph up` lokal

---

## Priorität

| Prio | Aufgabe | Warum |
|------|---------|-------|
| 1 | uv Setup & Dependencies | Grundlage für alles |
| 2 | MCP Server einzeln testen | Ohne das geht nichts |
| 3 | Env-Propagation | Kritischer Bug-Kandidat |
| 4 | Sync/Async Fix | Produktions-Stabilität |
| 5 | Multi-LLM | Nice-to-have |
| 6 | LangSmith | Monitoring |
| 7 | Integration Tests | Qualität |
| 8 | Security | Vor Production Deployment |
| 9 | Cleanup | Polish |

---

## Dateien die du kennen musst

| Datei | Beschreibung |
|-------|-------------|
| `scrumagent/agent.py` | Kern-Agent mit MCP Client |
| `mcp_servers/discord_chroma_server.py` | Custom MCP Server |
| `config/mcp_config.yaml` | MCP Server Konfiguration |
| `scrumagent/main_discord_bot.py` | Discord Bot (Einstiegspunkt) |
| `pyproject.toml` | Dependencies |
| `.env.example` | Env-Variablen Template |
| `tests/test_agent.py` | Unit Tests |
| `langgraph.json` | LangGraph Cloud Config |
