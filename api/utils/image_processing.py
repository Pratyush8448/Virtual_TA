import easyocr
import asyncio

reader = easyocr.Reader(['en'])

def _sync_ocr(image_path):
    result = reader.readtext(image_path, detail=0, paragraph=True)
    return " ".join(result)

async def extract_text_from_image(image_path: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_ocr, image_path)
