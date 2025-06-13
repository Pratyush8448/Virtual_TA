import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import base64
import os
import tempfile

from api.rag_model.generator import get_rag_chain
from api.rag_model.retriever import get_retriever

# Initialize app
app = FastAPI(
    title="Virtual TA - Tools in Data Science",
    version="1.0",
    description="An LLM-powered virtual teaching assistant for the TDS course by Pratyush Nishank"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up the RAG chain
retriever = get_retriever()
rag_chain = get_rag_chain(retriever)

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded image string

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link] = []

def save_base64_image(base64_str: str) -> str:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]  # Remove data URI prefix
        base64_bytes = base64_str.encode("utf-8")
        image_data = base64.b64decode(base64_bytes, validate=True)

        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "uploaded_image.webp")
        with open(file_path, "wb") as f:
            f.write(image_data)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


@app.post("/api/", response_model=QueryResponse)
async def process_question(req: QueryRequest):
    question = req.question
    image_data = req.image

    if image_data:
        try:
            image_path = save_base64_image(image_data)
            print(f"[DEBUG] Image saved at: {image_path}")
            from api.utils.image_processing import extract_text_from_image
            ocr_text = await extract_text_from_image(image_path)
            print(f"[OCR] Extracted Text: {ocr_text}")
            question = ocr_text + "\n\n" + question
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")

    response = await rag_chain.ainvoke({"question": question})
    if hasattr(response, "content"):
        response = response.content
    elif isinstance(response, dict) and "output" in response:
        response = response["output"]


    # Add links if present
    links = []
    if "GA5" in question:
        links.append(Link(
            url="https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939",
            text="GA5 Question 8 Clarification"
        ))

    return QueryResponse(answer=response, links=links)

@app.get("/")
def root():
    return {"message": "Virtual TA API is live 🚀"}
