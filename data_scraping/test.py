import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": embedding_model,
    "input": ["This is a test document for embedding."]
}

response = requests.post(f"{api_base}/embeddings", headers=headers, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
