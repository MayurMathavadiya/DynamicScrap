# Dynamic Scrap Agent 

`DynamicScrap` is an autonomous, LLM-driven web scraping agent built with Python, Playwright, BeautifulSoup, Pydantic, and the OpenAI SDK. It translates natural language instructions into browser actions, then extracts data into a user-defined schema without requiring hardcoded CSS or XPath selectors.

The repo currently provides two implementations:

- `DynamicScrap.py` — synchronous agent using `OpenAI` and `playwright.sync_api`.
- `DynamicScrapAsync.py` — asynchronous agent using `AsyncOpenAI` and `playwright.async_api`.

---

## Key Features

- **Autonomous Navigation**: Plans and runs Playwright actions such as click, type, hover, scroll, wait, pagination, and tab handling.
- **Search Handling**: Detects visible search inputs, types the query, and submits it.
- **Dynamic Extraction**: Generates extraction specs with LLM help and falls back to BeautifulSoup parsing.
- **Schema Mapping**: Maps extracted rows to your requested output schema.
- **Validation & Repair**: Checks required fields and proposes repair actions when page data is missing.
- **Sync + Async Support**: Use blocking scripts or async applications without changing task definitions.

---

## Tech Stack

- **Python**
- **Playwright** for browser automation
- **BeautifulSoup4** for HTML parsing
- **Pydantic** for task/schema validation
- **OpenAI SDK** for OpenAI-compatible model providers

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Configure these environment variables before running examples:

```bash
BASE_URL="https://api.groq.com/openai/v1"
API_KEY="your-api-key"
MODEL="your-model-name"
```

You can place them in a local `.env` file because the examples call `load_dotenv()`.

---

## Quick Start: Sync

Use `DynamicScrap.py` for normal blocking scripts.

```python
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from DynamicScrap import DynamicScrapeAgent, TaskSchema, ScrapeTask

load_dotenv()

def main() -> None:
    task_obj = ScrapeTask(
        url="https://news.ycombinator.com/",
        extract_instruction="Extract the top 5 stories with title and link.",
        output_schema=TaskSchema(
            items=[{"title": "string", "link": "string"}],
        ),
        required_fields=["title", "link"],
        max_pages=1,
        total_timeout_sec=120,
    )

    client = OpenAI(
        base_url=os.environ.get("BASE_URL"), # Remove if openAI
        api_key=os.environ.get("API_KEY"),
    )

    agent = DynamicScrapeAgent(
        model=os.environ.get("MODEL"),
        client=client,
        task=task_obj.model_dump(),
        headless=True,
    )

    try:
        result = agent.run_task()
        print(json.dumps(result["data"], indent=2, ensure_ascii=False))
        if not result["ok"]:
            print(json.dumps(result["diagnostics"], indent=2, ensure_ascii=False))
    finally:
        agent.close()

if __name__ == "__main__":
    main()
```

---

## Quick Start: Async

Use `DynamicScrapAsync.py` when your application already uses `asyncio`.

```python
import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from DynamicScrapAsync import DynamicScrapeAgent, TaskSchema, ScrapeTask

load_dotenv()

async def main() -> None:
    task_obj = ScrapeTask(
        url="https://news.ycombinator.com/",
        extract_instruction="Extract the top 5 stories with title and link.",
        output_schema=TaskSchema(
            items=[{"title": "string", "link": "string"}],
        ),
        required_fields=["title", "link"],
        max_pages=1,
        total_timeout_sec=120,
    )

    client = AsyncOpenAI(
        base_url=os.environ.get("BASE_URL"), # Remove if openAI
        api_key=os.environ.get("API_KEY"),
    )

    agent = DynamicScrapeAgent(
        model=os.environ.get("MODEL"),
        client=client,
        task=task_obj.model_dump(),
        headless=True,
    )

    try:
        result = await agent.run_task()
        print(json.dumps(result["data"], indent=2, ensure_ascii=False))
        if not result["ok"]:
            print(json.dumps(result["diagnostics"], indent=2, ensure_ascii=False))
    finally:
        await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Task Configuration

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `url` | `str` | Yes | - | Initial page URL. |
| `extract_instruction` | `str` | Yes | - | Natural language extraction instruction. |
| `output_schema` | `TaskSchema` | Yes | - | Expected output fields and types. |
| `search_query` | `str` | No | `None` | Query to submit through the page search box. |
| `action_instruction` | `str` | No | `None` | Browser action instruction before extraction. |
| `required_fields` | `List[str]` | No | `None` | Fields required for validation success. |
| `max_pages` | `int` | No | `1` | Number of pages to collect through pagination. |
| `max_repairs` | `int` | No | `2` | Repair retries when source data is missing. |
| `total_timeout_sec` | `int` | No | `300` | Overall task timeout in seconds. |

---

## Example Scripts

### Sync Examples

```bash
python3 example_github_scrap.py
python3 example_amazon_scrap.py
```

- `example_github_scrap.py` demonstrates hover navigation on GitHub.
- `example_amazon_scrap.py` demonstrates search, extraction, pagination, and validation.

### Async Examples

```bash
python3 example_github_scrap_async.py
python3 example_amazon_scrap_async.py
```

- `example_github_scrap_async.py` is the async version of the GitHub workflow.
- `example_amazon_scrap_async.py` is the async version of the Amazon workflow.

---

## Model Providers

The agents use the OpenAI SDK interface, so any OpenAI-compatible provider can work if it supports chat completions.

### Examples:


### 1. OpenAI (ChatGPT)

> ⚠️ **Important**
>
> Do **not** pass the `base_url` parameter when using OpenAI directly.
>
> The OpenAI SDK automatically uses the official OpenAI API endpoint. Passing a custom `base_url` value (including `""` or `None`) may result in connection or initialization errors.

**✅ Correct**

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")
```

**❌ Incorrect**

```python
client = OpenAI(
    api_key="your-api-key",
    base_url=""
)
```

```python
client = OpenAI(
    api_key="your-api-key",
    base_url=None
)
```

---

### 2. Together

```bash
BASE_URL="https://api.together.xyz/v1"
API_KEY="your-together-api-key"
MODEL="meta-llama/Llama-3-70b-chat-hf"
```

---

### 3. OpenRouter

```bash
BASE_URL="https://openrouter.ai/api/v1"
API_KEY="your-openrouter-key"
MODEL="anthropic/claude-3.5-sonnet"
```

---

### 4. Ollama (Locally)

```bash
BASE_URL="http://localhost:11434/v1"
API_KEY="ollama"
MODEL="qwen2.5:7b"
```
