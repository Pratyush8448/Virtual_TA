# openai_client.py
import os
from openai import OpenAI

def get_openai_client():
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://aiproxy.sanand.workers.dev/v1")  # default if not set
    )
