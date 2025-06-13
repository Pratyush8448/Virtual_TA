# File: api/rag_model/retriever.py

import os
import requests
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda


CHUNKS_FOLDER = "data_chunks"
INDEX_PATH = "faiss_index"


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
                raise Exception(f"[Embedding Error] Document #{i} failed: {response.status_code} {response.text}")

            embedding_vector = response.json()["data"][0]["embedding"]
            embeddings.append(embedding_vector)

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
    def __init__(self, chunks_folder=CHUNKS_FOLDER, index_path=INDEX_PATH):
        self.chunks_folder = chunks_folder
        self.index_path = index_path

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        if not self.api_key or not self.api_base:
            raise ValueError("Environment variables OPENAI_API_KEY and OPENAI_API_BASE must be set.")

        self.embeddings = ProxyEmbeddings(api_key=self.api_key, api_base=self.api_base, model=self.model)
        self.index = self.load_or_build_index()

    def load_or_build_index(self) -> FAISS:
        if os.path.exists(self.index_path):
            print(f"[Retriever] Loading existing FAISS index from {self.index_path}...")
            return FAISS.load_local(
                self.index_path, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            print(f"[Retriever] Building new FAISS index from folder: {self.chunks_folder}")
            return self.build_index()

    def build_index(self) -> FAISS:
        texts, metadatas = [], []
        files = [f for f in os.listdir(self.chunks_folder) if f.endswith(".txt")]

        for filename in files:
            path = os.path.join(self.chunks_folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                metadatas.append({"source": filename})

        documents = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        index = FAISS.from_documents(documents, self.embeddings)
        index.save_local(self.index_path)
        print(f"[Retriever] FAISS index saved at {self.index_path}")
        return index

    def get_vectorstore_retriever(self):
        return self.index.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def retrieve(self, query: str) -> List[Document]:
        retriever = self.get_vectorstore_retriever()
        return retriever.get_relevant_documents(query)


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
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs):
        print(f"\nResult {i + 1}:\n{doc.page_content[:500]}...\n")
