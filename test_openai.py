#Retriever.py
import os
import json
import requests
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda

CHUNKS_FOLDER = "data_chunks"
DISCOURSE_FOLDER = "downloaded_threads"
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
    def __init__(self, chunks_folder=CHUNKS_FOLDER, index_path=INDEX_PATH, discourse_folder=DISCOURSE_FOLDER):
        self.chunks_folder = chunks_folder
        self.index_path = index_path
        self.discourse_folder = discourse_folder

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
            print(f"[Retriever] Building new FAISS index from text chunks and Discourse threads...")
            return self.build_index()

    def build_index(self) -> FAISS:
        documents = []

        # Load plain .txt chunks
        txt_files = [f for f in os.listdir(self.chunks_folder) if f.endswith(".txt")]
        for filename in txt_files:
            path = os.path.join(self.chunks_folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))

        # Load Discourse threads
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
        print(f"[Retriever] FAISS index built and saved at {self.index_path}")
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








#generator.py
import os
#from api.openai_client import client

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from api.utils.image_processing import extract_text_from_image
from api.rag_model.retriever import get_retriever


def get_rag_chain(retriever):
    print("[DEBUG] Setting up ChatOpenAI model...")

    llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENAI_API_BASE")  # ✅ use this
)

    print("[DEBUG] Defining RAG prompt...")
    prompt = ChatPromptTemplate.from_template(
        "You are a TA for the Tools in Data Science course.\n"
        "Use the following context to answer the question.\n\n"
        "{context}\n\n"
        "Question: {question}"
    )

    def preprocess_inputs(inputs):
        question = inputs.get("question", "")
        image_path = inputs.get("image_path")
        if image_path:
            try:
                extracted_text = extract_text_from_image(image_path)
                print(f"[INFO] OCR Text from {image_path}:\n{extracted_text}")
                question += f"\n\n[Image Text]: {extracted_text}"
            except Exception as e:
                print(f"[ERROR] Failed to extract text from image: {e}")
        return {"question": question}

    print("[DEBUG] Assembling RAG chain...")
    rag_chain = (
        {"context": retriever, "question": lambda x: preprocess_inputs(x)["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
# Optional: For standalone testing
if __name__ == "__main__":
    import asyncio

    async def run_chain():
        print("[DEBUG] Loading retriever...")
        retriever = get_retriever()

        print("[DEBUG] Initializing RAG chain...")
        rag_chain = get_rag_chain(retriever)

        test_question = "What is a confusion matrix?"
        print(f"[DEBUG] Running RAG chain on question: {test_question}")
        result = await rag_chain.ainvoke({"question": test_question})
        print("[DEBUG] Output:\n", result)

    asyncio.run(run_chain())


#mangum>=0.17.0
