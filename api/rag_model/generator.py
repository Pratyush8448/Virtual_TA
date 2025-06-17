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
    openai_api_base=os.environ.get("OPENAI_API_BASE") 
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
