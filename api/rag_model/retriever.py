import os
import json
import requests
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.runnables import RunnableLambda


class ProxyEmbeddings(Embeddings):
    def __init__(self, api_key: str, api_base: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"[ERROR] Input at index {i} is not a string. Got: {type(text)}")
            payload = {"model": self.model, "input": text}
            response = requests.post(f"{self.api_base}/embeddings", headers=self.headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"[Embedding Error] Doc #{i} failed: {response.status_code} {response.text}")
            embeddings.append(response.json()["data"][0]["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        if not isinstance(text, str):
            raise ValueError(f"[ERROR] Embedding input must be a string. Got: {type(text)}")
        payload = {"model": self.model, "input": text}
        response = requests.post(f"{self.api_base}/embeddings", headers=self.headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Embedding query failed: {response.status_code} {response.text}")
        return response.json()["data"][0]["embedding"]


class Retriever:
    def __init__(self):
        self.embeddings = ProxyEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )
        self.index_path = "faiss_index"
        self.chunks_folder = "data_chunks"
        self.discourse_folder = "downloaded_threads"
        self.index = self.load_or_build_index()

    def load_or_build_index(self):
        index_faiss_path = os.path.join(self.index_path, "index.faiss")
        index_pkl_path = os.path.join(self.index_path, "index.pkl")

        if not os.path.exists(index_faiss_path) or not os.path.exists(index_pkl_path):
            print("[INFO] No FAISS index found. Rebuilding from scratch.")
            return self.build_index()

        try:
            print("[INFO] Loading FAISS index from disk...")
            return FAISS.load_local(
                folder_path=self.index_path,
                embeddings=self.embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to load FAISS index: {e}")
            print("[INFO] Rebuilding FAISS index instead...")
            return self.build_index()

    def build_index(self) -> FAISS:
        documents = []

        # Load .txt chunks
        if os.path.exists(self.chunks_folder):
            txt_files = [f for f in os.listdir(self.chunks_folder) if f.endswith(".txt")]
            for filename in txt_files:
                path = os.path.join(self.chunks_folder, filename)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(Document(page_content=content, metadata={"source": filename}))

        # Load Discourse threads
        if os.path.exists(self.discourse_folder):
            json_files = [f for f in os.listdir(self.discourse_folder) if f.endswith(".json")]
            for filename in json_files:
                path = os.path.join(self.discourse_folder, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        title = data.get("title", "")
                        posts = data.get("post_stream", {}).get("posts", [])
                        for post in posts:
                            body = post.get("cooked", "")
                            text = f"{title}\n{body}".strip()
                            if text:
                                documents.append(Document(page_content=text, metadata={"source": filename}))
                except Exception as e:
                    print(f"[WARNING] Failed to load {filename}: {e}")

        index = FAISS.from_documents(documents, self.embeddings)
        index.save_local(self.index_path)
        print(f"[Retriever] FAISS index built and saved at '{self.index_path}'")
        return index

    def get_vectorstore_retriever(self):
        return self.index.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def retrieve(self, query: str) -> List[Document]:
        retriever = self.get_vectorstore_retriever()
        raw_docs = retriever.get_relevant_documents(query)

        # Deduplicate results by metadata["source"]
        seen_sources = set()
        unique_docs = []
        for doc in raw_docs:
            src = doc.metadata.get("source", "")
            if src not in seen_sources:
                seen_sources.add(src)
                unique_docs.append(doc)
            if len(unique_docs) >= 3:  # Return top 3 unique sources
                break
        return unique_docs


def get_retriever():
    vector_retriever = Retriever().get_vectorstore_retriever()
    return RunnableLambda(
        lambda x: [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vector_retriever.invoke(x["question"])
        ]
    )


if __name__ == "__main__":
    retriever = get_retriever()
    query = input("Enter your query: ").strip()
    docs = retriever.invoke({"question": query})
    for i, doc in enumerate(docs):
        print(f"\nResult {i + 1}:\n{doc.page_content[:500]}...\n")
