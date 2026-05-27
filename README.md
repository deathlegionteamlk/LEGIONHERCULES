# LEGIONHERCULES

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg" alt="Ollama">
  <img src="https://img.shields.io/badge/Async-Parallel%20Agents-purple.svg" alt="Async">
</p>

<p align="center">
  <b>Autonomous CLI Framework with Parallel Agent Execution</b>
</p>

<p align="center">
  <i>Developed by Death Legion Team Coders Demo X HEXA</i>
</p>

---

## 🚀 What is LEGIONHERCULES?

LEGIONHERCULES is a powerful, open-source CLI framework that brings autonomous AI agents to your terminal. Built for developers who want the power of Claude Code or OpenCode, but **completely free** and running **locally** on their own hardware.

### ✨ Key Features

- 🤖 **Autonomous Agents** - AI agents that can reason, plan, and execute tasks
- ⚡ **Parallel Execution** - Run multiple agents simultaneously for faster results
- 🔧 **Built-in Tools** - File operations, bash commands, web search out of the box
- 🆓 **Completely Free** - Uses Ollama for local LLM inference - no API keys needed
- 💻 **Local-First** - All processing happens on your machine
- 🎨 **Rich Terminal UI** - Beautiful, interactive CLI experience

---

## 📦 Installation

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

## 🚀 Quick Start

### 1. Start Ollama

```bash
ollama serve
```

### 2. Pull a Model

```bash
ollama pull llama3.2
```

### 3. Start Chatting

```bash
# Interactive mode
legionhercules chat

# Or send a single message
legionhercules chat "Hello, what can you do?"
```

---

## 📖 Usage Examples

### Interactive Chat

```bash
legionhercules chat
```

### Using Different Models

```bash
legionhercules chat --model codellama
legionhercules chat --model mistral
```

### Using Different Agents

```bash
# Code-focused agent
legionhercules chat --agent coder

# Research agent with web search
legionhercules chat --agent researcher
```

### Managing Models

```bash
# List available models
legionhercules models --list

# Pull a new model
legionhercules models --pull llama3.1
```

---

## 🛠️ Available Tools

LEGIONHERCULES comes with powerful built-in tools:

| Tool | Description |
|------|-------------|
| `file_read` | Read file contents with optional offset and limit |
| `file_write` | Write or append to files |
| `file_edit` | Search and replace text in files |
| `bash` | Execute bash commands with safety checks |
| `web_search` | Search the web using DuckDuckGo |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│              CLI Layer                │
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

## 📚 Documentation

- [Installation Guide](https://legionhercules.dev/installation)
- [Quick Start](https://legionhercules.dev/quickstart)
- [Agents](https://legionhercules.dev/agents)
- [Tools](https://legionhercules.dev/tools)
- [Configuration](https://legionhercules.dev/configuration)
- [API Reference](https://legionhercules.dev/api)

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://legionhercules.dev/contributing) for details.

### Quick Development Setup

```bash
git clone https://github.com/deathlegion/legionhercules.git
cd legionhercules
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com) - For making local LLMs accessible
- [Rich](https://github.com/Textualize/rich) - For the beautiful terminal UI
- [Typer](https://typer.tiangolo.com) - For the CLI framework

---

## 📄 License

LEGIONHERCULES is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Developed with ❤️ by Death Legion Team Coders Demo X HEXA</b>
</p>

<p align="center">
  <a href="https://legionhercules.dev">Website</a> •
  <a href="https://github.com/deathlegion/legionhercules">GitHub</a> •
  <a href="https://legionhercules.dev/docs">Documentation</a>
</p>
