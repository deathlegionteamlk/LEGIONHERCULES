<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&pause=1000&color=A855F7&center=true&vCenter=true&width=600&lines=LEGIONHERCULES;Autonomous+CLI+Framework;Parallel+Agent+Execution" alt="LEGIONHERCULES Typing Animation" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg" alt="Ollama">
  <img src="https://img.shields.io/badge/Async-Parallel%20Agents-purple.svg" alt="Async">
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=16&pause=1000&color=6B7280&center=true&vCenter=true&width=500&lines=Developed+by+Death+Legion+Team+Coders+Demo+X+HEXA" alt="Team Typing" />
</p>

---

## <img src="https://img.icons8.com/fluency/28/launch.png" width="22"/> What is LEGIONHERCULES?

LEGIONHERCULES is an open-source CLI framework that runs autonomous AI agents in your terminal. It's built for developers who want something like Claude Code or OpenCode — but free, and running entirely on their own hardware.

No subscriptions. No API keys. Just Ollama under the hood and your machine doing the work.

---

## <img src="https://img.icons8.com/fluency/28/star.png" width="22"/> Key Features

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&pause=800&color=10B981&center=true&vCenter=true&multiline=true&width=500&height=100&lines=Autonomous+Agents;Parallel+Execution;Built-in+Tools;Completely+Free+%26+Local-First" alt="Features Typing" />
</p>

| | Feature | What it means |
|---|---|---|
| <img src="https://img.icons8.com/fluency/20/robot.png"/> | **Autonomous Agents** | Agents that reason, plan, and execute — not just respond |
| <img src="https://img.icons8.com/fluency/20/lightning-bolt.png"/> | **Parallel Execution** | Multiple agents running at once, not waiting in line |
| <img src="https://img.icons8.com/fluency/20/settings.png"/> | **Built-in Tools** | File ops, bash, and web search included by default |
| <img src="https://img.icons8.com/fluency/20/price-tag.png"/> | **Free** | Ollama handles inference locally — no external billing |
| <img src="https://img.icons8.com/fluency/20/computer.png"/> | **Local-First** | Everything stays on your machine |
| <img src="https://img.icons8.com/fluency/20/paint-palette.png"/> | **Rich Terminal UI** | Looks good in the terminal, actually |

---

## <img src="https://img.icons8.com/fluency/28/box.png" width="22"/> Installation

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.com) installed and running

### Install from PyPI

```bash
pip install legionhercules
```

### Install from Source

```bash
git clone https://github.com/deathlegion/legionhercules.git
cd legionhercules
pip install -e ".[dev]"
```

---

## <img src="https://img.icons8.com/fluency/28/launch.png" width="22"/> Quick Start

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&pause=1000&color=F59E0B&center=true&vCenter=true&width=450&lines=1.+Start+Ollama;2.+Pull+a+Model;3.+Start+Chatting" alt="Quick Start Steps" />
</p>

**1. Start Ollama**

```bash
ollama serve
```

**2. Pull a model**

```bash
ollama pull llama3.2
```

**3. Start chatting**

```bash
# Interactive mode
legionhercules chat

# Or send a one-off message
legionhercules chat "Hello, what can you do?"
```

---

## <img src="https://img.icons8.com/fluency/28/book.png" width="22"/> Usage Examples

### Interactive Chat

```bash
legionhercules chat
```

### Different Models

```bash
legionhercules chat --model codellama
legionhercules chat --model mistral
```

### Different Agents

```bash
# Code-focused agent
legionhercules chat --agent coder

# Research agent with web search
legionhercules chat --agent researcher
```

### Model Management

```bash
# List available models
legionhercules models --list

# Pull a new model
legionhercules models --pull llama3.1
```

---

## <img src="https://img.icons8.com/fluency/28/wrench.png" width="22"/> Built-in Tools

| Tool | What it does |
|------|-------------|
| `file_read` | Read file contents, with optional offset and line limit |
| `file_write` | Write or append to files |
| `file_edit` | Find-and-replace inside files |
| `bash` | Run bash commands (with safety checks) |
| `web_search` | Search the web via DuckDuckGo |

---

## <img src="https://img.icons8.com/fluency/28/blueprint.png" width="22"/> Architecture

```
┌─────────────────────────────────────────┐
│              CLI Layer                  │
│         (Rich Terminal UI)              │
├─────────────────────────────────────────┤
│           Orchestrator                  │
│    (Parallel Task Management)           │
├─────────────────────────────────────────┤
│            Agents                       │
│  (Reasoning + Tool Use + LLM)           │
├─────────────────────────────────────────┤
│            Tools                        │
│  (File, Bash, Web Search)               │
├─────────────────────────────────────────┤
│         LLM Provider                    │
│       (Ollama Integration)              │
└─────────────────────────────────────────┘
```

---

## <img src="https://img.icons8.com/fluency/28/documents.png" width="22"/> Documentation

- [Installation Guide](https://legionhercules.dev/installation)
- [Quick Start](https://legionhercules.dev/quickstart)
- [Agents](https://legionhercules.dev/agents)
- [Tools](https://legionhercules.dev/tools)
- [Configuration](https://legionhercules.dev/configuration)
- [API Reference](https://legionhercules.dev/api)

---

## <img src="https://img.icons8.com/fluency/28/handshake.png" width="22"/> Contributing

Pull requests are welcome. Check the [Contributing Guide](https://legionhercules.dev/contributing) before you start.

### Quick Dev Setup

```bash
git clone https://github.com/deathlegion/legionhercules.git
cd legionhercules
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## <img src="https://img.icons8.com/fluency/28/gratitude.png" width="22"/> Credits

- [Ollama](https://ollama.com) — local LLM inference that doesn't require a cloud account
- [Rich](https://github.com/Textualize/rich) — terminal UI that doesn't look like 1995
- [Typer](https://typer.tiangolo.com) — the CLI backbone

---

## <img src="https://img.icons8.com/fluency/28/document.png" width="22"/> License

MIT. See [LICENSE](LICENSE) for the full text.

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&pause=1000&color=EF4444&center=true&vCenter=true&width=500&lines=Developed+with+%E2%9D%A4%EF%B8%8F+by+Death+Legion+Team+Coders+Demo+X+HEXA" alt="Footer Typing" />
</p>

<p align="center">
  <a href="https://legionhercules.dev">Website</a> •
  <a href="https://github.com/deathlegion/legionhercules">GitHub</a> •
  <a href="https://legionhercules.dev/docs">Documentation</a>
</p>
