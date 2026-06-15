import os
import json
from openai import OpenAI
from dotenv import load_dotenv


from DynamicScrap import DynamicScrapeAgent, TaskSchema, ScrapeTask


load_dotenv()


def main() -> None:
    # Task definition
    task_obj = ScrapeTask(
        url="https://github.com/",
        search_query="",
        action_instruction="Hover on Solutions in the header.",
        extract_instruction="Extract the solutions provide by github.com, like by company, by use case and etc.",
        output_schema=TaskSchema(
            items=[
                {
                    "name": "string",
                    "url": "string",
                }
            ],
        ),
        required_fields=["name", "url"],
        max_pages=1,
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
