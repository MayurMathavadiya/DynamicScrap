<div align="center">

# 🤖 DynamicScrap

### Autonomous LLM-Powered Web Scraping Agent

Turn natural-language instructions into browser actions and structured web data.

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Playwright-Browser%20Automation-2EAD33?style=for-the-badge&logo=playwright&logoColor=white" alt="Playwright Browser Automation">
  <img src="https://img.shields.io/badge/LLM-Powered-8A2BE2?style=for-the-badge" alt="LLM Powered">
  <img src="https://img.shields.io/badge/License-Open%20Source-blue?style=for-the-badge" alt="Open Source">
</p>

</div>

---

<p align="center">
  <img src="./DynamicScrap.png" alt="DynamicScrap Architecture" width="100%">
</p>

<p align="center">
  <b>Natural Language → LLM Agent → Browser Automation → Dynamic Extraction → Structured Data</b>
</p>

---

## 🚀 What is DynamicScrap?

**DynamicScrap** is an autonomous web scraping framework that uses **LLMs + Playwright + BeautifulSoup** to understand websites, navigate pages, perform browser actions, and extract structured information.

Instead of writing brittle selectors like:

```python
id="traditional-example"
driver.find_element(By.CSS_SELECTOR, ".product-card .title")
```

you can describe what you need:

```text
Extract the top 10 products with:
- product name
- price
- rating
- product URL
```

DynamicScrap allows the LLM agent to determine **how to navigate the website and locate the requested information**.

---

## 🧠 The Idea

### Traditional Scraping

```text
Website
   ↓
CSS / XPath Selectors
   ↓
HTML Elements
   ↓
Extract Data
```

Website changes its HTML?

```text
❌ Selector breaks
❌ Scraper breaks
❌ Developer updates selectors
```

### DynamicScrap

```text
Natural Language Task
        ↓
     LLM Agent
        ↓
 Understand Website
        ↓
 Browser Navigation
        ↓
 Dynamic Extraction
        ↓
 Validation & Repair
        ↓
 Structured JSON
```

The goal is to move from:

> **"Tell the scraper where the data is."**

to:

> **"Tell the agent what data you need."**

---

# ✨ Key Features

| Feature               | Description                                           |
| --------------------- | ----------------------------------------------------- |
| 🧠 LLM Agent          | Uses an LLM for planning and browser decisions        |
| 🌐 Browser Automation | Real browser interaction through Playwright           |
| 🔎 Dynamic Search     | Detects and interacts with search inputs              |
| 🖱️ Browser Actions   | Click, type, hover, scroll, wait and navigation       |
| 📄 Dynamic Extraction | Extracts information without fixed selectors          |
| 📦 Structured Output  | Returns data according to a defined schema            |
| ✅ Validation          | Checks required fields in extracted data              |
| 🔧 Repair             | Attempts to recover missing information               |
| 📑 Pagination         | Supports multi-page scraping workflows                |
| ⚡ Async Support       | Includes synchronous and asynchronous implementations |
| 🤖 Multiple LLMs      | Works with OpenAI-compatible providers                |
| 🏠 Local Models       | Can be configured with Ollama                         |

---

# 🎯 Example

Imagine you want information from an e-commerce website.

Instead of manually writing selectors for every element, define the task:

```python
task = ScrapeTask(
    url="https://example.com",
    extract_instruction="""
        Extract the first 10 products with:
        - product name
        - price
        - rating
        - product URL
    """,
    output_schema=TaskSchema(
        items=[
            {
                "name": "string",
                "price": "string",
                "rating": "string",
                "url": "string",
            }
        ]
    ),
    required_fields=[
        "name",
        "price",
        "rating",
        "url",
    ],
)
```

The agent can then:

```text
Open Website
     ↓
Understand Page
     ↓
Find Relevant Elements
     ↓
Interact With Page
     ↓
Extract Products
     ↓
Validate Required Fields
     ↓
Return Structured Data
```

---

# 🏗️ Architecture

<p align="center">
  <img src="./DynamicScrap.png" alt="DynamicScrap System Architecture" width="100%">
</p>

### High-Level Flow

```text
┌──────────────────────────────┐
│     Natural Language Task    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          LLM Agent           │
│   Planning / Decision Layer  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          Playwright          │
│      Browser Automation      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│         Target Website       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Dynamic Extraction       │
│     BeautifulSoup + LLM      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Validation & Repair     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Structured Output      │
│             JSON             │
└──────────────────────────────┘
```

---

# 🛠️ Tech Stack

<p align="center">

**Python** · **Playwright** · **BeautifulSoup4** · **Pydantic** · **OpenAI SDK** · **Ollama**

</p>

### Core Technologies

* 🐍 Python
* 🎭 Playwright
* 🍲 BeautifulSoup4
* 🧱 Pydantic
* 🤖 OpenAI SDK
* 🧠 OpenAI-compatible APIs
* 🏠 Ollama

---

# ⚡ Quick Start

## 1. Clone

```bash
git clone https://github.com/MayurMathavadiya/DynamicScrap.git

cd DynamicScrap
```

## 2. Create Environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install Playwright Browser

```bash
playwright install chromium
```

---

# 🔐 Configuration

Create a `.env` file:

```env
BASE_URL="https://api.groq.com/openai/v1"
API_KEY="your-api-key"
MODEL="your-model"
```

> ⚠️ Never commit your API keys or `.env` file.

---

# 💻 Basic Usage

```python
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from DynamicScrap import (
    DynamicScrapeAgent,
    TaskSchema,
    ScrapeTask,
)

load_dotenv()

task = ScrapeTask(
    url="https://news.ycombinator.com/",
    extract_instruction="Extract the top 5 stories with title and link.",
    output_schema=TaskSchema(
        items=[
            {
                "title": "string",
                "link": "string",
            }
        ]
    ),
    required_fields=["title", "link"],
    max_pages=1,
    total_timeout_sec=120,
)

client = OpenAI(
    base_url=os.environ.get("BASE_URL"),
    api_key=os.environ.get("API_KEY"),
)

agent = DynamicScrapeAgent(
    model=os.environ.get("MODEL"),
    client=client,
    task=task.model_dump(),
    headless=True,
)

try:
    result = agent.run_task()

    print(
        json.dumps(
            result["data"],
            indent=2,
            ensure_ascii=False,
        )
    )

finally:
    agent.close()
```

---

# ⚡ Sync & Async

DynamicScrap provides both execution styles.

### Synchronous

```python
from DynamicScrap import DynamicScrapeAgent
```

### Asynchronous

```python
from DynamicScrapAsync import DynamicScrapeAgent
```

This makes the framework suitable for both traditional Python scripts and applications built around `asyncio`.

---

# 🤖 LLM Provider Support

DynamicScrap uses the OpenAI SDK interface, allowing OpenAI-compatible providers to be configured through `BASE_URL`.

### OpenAI

```env
API_KEY="your-api-key"
MODEL="your-model"
```

### Groq

```env
BASE_URL="https://api.groq.com/openai/v1"
API_KEY="your-api-key"
MODEL="your-model"
```

### OpenRouter

```env
BASE_URL="https://openrouter.ai/api/v1"
API_KEY="your-api-key"
MODEL="your-model"
```

### Together AI

```env
BASE_URL="https://api.together.xyz/v1"
API_KEY="your-api-key"
MODEL="your-model"
```

### Ollama

```env
BASE_URL="http://localhost:11434/v1"
API_KEY="ollama"
MODEL="qwen2.5:7b"
```

---

# 🧪 Examples

## GitHub

```bash
python3 example_github_scrap.py
```

Demonstrates browser navigation and interaction.

## Amazon

```bash
python3 example_amazon_scrap.py
```

Demonstrates:

* Search
* Dynamic extraction
* Pagination
* Validation

## Async

```bash
python3 example_github_scrap_async.py

python3 example_amazon_scrap_async.py
```

---

# 📁 Project Structure

```text
DynamicScrap/
│
├── DynamicScrap.py
├── DynamicScrapAsync.py
│
├── example_github_scrap.py
├── example_github_scrap_async.py
│
├── example_amazon_scrap.py
├── example_amazon_scrap_async.py
│
├── requirements.txt
├── DynamicScrap.png
└── README.md
```

---

# 🔥 Why This Project?

DynamicScrap explores an important direction in modern web automation:

```text
Traditional Automation
        ↓
Hardcoded Selectors
        ↓
Fixed Workflows
```

versus:

```text
AI-Powered Automation
        ↓
Natural Language Instructions
        ↓
Dynamic Decisions
        ↓
Adaptive Browser Workflows
```

The project combines **browser automation, LLM reasoning, structured extraction, validation, and recovery** into a single scraping workflow.

---

# 🗺️ Roadmap

* [ ] Not decided yet 😂. Please raise issue if you have new idea 💡.

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes
5. Open a Pull Request

---

# ⚠️ Responsible Use

DynamicScrap automates browser interaction and web data extraction.

Users are responsible for ensuring their scraping activities comply with:

* Website Terms of Service
* `robots.txt`
* Applicable laws
* Data privacy requirements
* Copyright requirements

Please use the project responsibly.

---

# ⭐ Support

If you find DynamicScrap useful:

⭐ Star the repository
🍴 Fork the project
🐛 Report issues
💡 Suggest features
🤝 Contribute

---

## 📄 License

Free for all. Enjoy forks 😁

---

## Star History

<a href="https://www.star-history.com/?type=date&repos=MayurMathavadiya%2FDynamicScrap">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=MayurMathavadiya/DynamicScrap&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=MayurMathavadiya/DynamicScrap&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=MayurMathavadiya/DynamicScrap&type=date&legend=top-left" />
 </picture>
</a>
