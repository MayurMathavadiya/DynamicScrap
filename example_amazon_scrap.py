import os
import json
from openai import OpenAI
from dotenv import load_dotenv


from DynamicScrap import DynamicScrapeAgent, TaskSchema, ScrapeTask


load_dotenv()


def main() -> None:
    # Task definition
    task_obj = ScrapeTask(
        url="https://www.amazon.in/",
        search_query="wireless headphones",
        action_instruction="Search query and open results.",
        extract_instruction="Extract 10 products with title, price, rating, and product url.",
        output_schema=TaskSchema(
            items={
                "title": "string",
                "price": "string",
                "rating": "string",
                "product_url": "string",
            },
        ),
        required_fields=["title", "price"],
        max_pages=2,
        max_repairs=2,
        total_timeout_sec=480,
    )

    # Convert task_obj to dict
    task = task_obj.model_dump()

    # LLM client config
    client = OpenAI(
        base_url=os.environ.get("BASE_URL"),
        api_key=os.environ.get("API_KEY"),
    )

    # Agent initialization
    agent = DynamicScrapeAgent(
        model=os.environ.get("MODEL"),
        client=client,
        task=task,
        headless=True,
    )

    # Execution
    try:
        # Run task
        result = agent.run_task()

        # Result printing
        print("\nRESULT:\n")
        print(json.dumps(
            result["data"], 
            indent=2, 
            ensure_ascii=False
        ))

        # Diagnostics printing in case of failure
        if not result["ok"]:
            print("\nDIAGNOSTICS:\n")
            print(json.dumps(
                result["diagnostics"], 
                indent=2, 
                ensure_ascii=False
            ))

    finally:
        # Close browser
        agent.close()


if __name__ == "__main__":
    main()
