import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import glob

def load_discourse_threads():
    thread_data = []
    for file in glob.glob("downloaded_threads/*.json"):
        with open(file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                thread_data.extend(data.get("post_stream", {}).get("posts", []))
            except:
                continue
    return thread_data

discourse_threads = load_discourse_threads()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import base64
import tempfile
import json

from api.rag_model.generator import get_rag_chain
from api.rag_model.retriever import get_retriever

# Load discourse thread metadata
with open("data_chunks/discourse_threads.json", "r", encoding="utf-8") as f:
    discourse_threads = json.load(f)

# Initialize FastAPI app
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

# Set up RAG pipeline
retriever = get_retriever()
rag_chain = get_rag_chain(retriever)

# Request and response models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded image

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link] = []

# Function to decode base64 image
def save_base64_image(base64_str: str) -> str:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        base64_bytes = base64_str.encode("utf-8")
        image_data = base64.b64decode(base64_bytes, validate=True)

        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "uploaded_image.webp")
        with open(file_path, "wb") as f:
            f.write(image_data)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

from difflib import get_close_matches  # Add this at the top if not already

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

    # Smart matching with cooked post content
    matched_links = []
    question_words = set(question.lower().split())

    for post in discourse_threads:
        if "cooked" in post:
            cooked = post["cooked"].lower()
            if any(word in cooked for word in question_words):
                slug = post.get("topic_slug", "")
                topic_id = post.get("topic_id", "")
                if slug and topic_id:
                    url = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}"
                    title = slug.replace("-", " ").title()
                    if url not in [link.url for link in matched_links]:  # avoid duplicates
                        matched_links.append(Link(url=url, text=title))
        if len(matched_links) >= 3:
            break

    # Always include the main forum link at the top
    matched_links.insert(0, Link(
        url="https://discourse.onlinedegree.iitm.ac.in/",
        text="Visit the IITM Discourse Forum for related discussions"
    ))

    return QueryResponse(answer=response, links=matched_links)


@app.get("/")
def root():
    return {"message": "Virtual TA API is live 🚀"}
