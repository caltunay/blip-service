from fastapi import FastAPI
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import io
from pydantic import BaseModel
import numpy as np
from translate import translate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

class ImageURL(BaseModel):
    url: str
    text: str

blip_processor: BlipProcessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # type: ignore
blip_model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")  # type: ignore
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.post("/embed")
def embed_image(request: ImageURL):
    response = requests.get(request.url)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert('RGB')
    inputs = blip_processor(image, return_tensors="pt")  # type: ignore
    out = blip_model.generate(**inputs, max_length=50)  # type: ignore
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    text_translated = translate(text=request.text, source="TR", target="EN")
    caption_embedding = embedding_model.encode(caption)
    text_embedding = embedding_model.encode(text_translated)

    cosine_similarity = np.dot(caption_embedding, text_embedding) / (np.linalg.norm(caption_embedding) * np.linalg.norm(text_embedding))
    caption_translated = translate(text=caption, source="EN", target="TR")  
    return {"caption": caption_translated, "text": request.text, "cosine_similarity": float(cosine_similarity)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
