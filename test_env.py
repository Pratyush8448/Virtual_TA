from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import base64
import os
import tempfile
import shutil

from api.rag_model.generator import get_rag_chain
from api.rag_model.retriever import get_retriever
from api.utils.image_processing import extract_text_from_image

# Initialize app
app = FastAPI(
    title="Virtual TA - Tools in Data Science",
    version="1.0",
    description="An LLM-powered virtual teaching assistant for the TDS course by Pratyush Nishank"
)

# CORS (allow everything for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG setup
retriever = get_retriever()
rag_chain = get_rag_chain(retriever)

# Request & Response Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded image

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link] = []

# Helper to save base64 image to temp file
def save_base64_image(base64_str: str) -> str:
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "uploaded_image.webp")
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(base64_str))
        return file_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

# POST endpoint
@app.post("/api/", response_model=QueryResponse)
async def process_question(req: QueryRequest):
    question = req.question
    image_data = req.image

    # OCR if image is present
    if image_data:
        try:
            image_path = save_base64_image(image_data)
            print(f"[DEBUG] Image saved at: {image_path}")
            ocr_text = await extract_text_from_image(image_path)
            print(f"[OCR] Extracted Text: {ocr_text}")
            question = ocr_text + "\n\n" + question
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")

    # Get answer
    response = await rag_chain.ainvoke({"question": question})

    # Hardcoded link logic
    links = []
    if "GA5" in question:
        links.append({
            "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939",
            "text": "GA5 Question 8 Clarification"
        })

    return {"answer": response, "links": links}

# Health check
@app.get("/")
def root():
    return {"message": "Virtual TA API is live 🚀"}
