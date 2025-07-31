import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

model = "gemini-2.0-flash"

# 👇 Using direct API key as requested
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

def build_content(image_bytes: bytes, user_text: Optional[str]):
    parts = [types.Part.from_bytes(mime_type="image/png", data=image_bytes)]
    if user_text:
        parts.append(types.Part(text=user_text))
    return [types.Content(role="user", parts=parts)]

@app.post("/extract-json")
async def extract_json(
    system_instruction: str = Form(...),
    user_text: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    try:
        image_bytes = await image.read()
        contents = build_content(image_bytes, user_text)

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            system_instruction=[
                types.Part(text=system_instruction)
            ],
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        data = json.loads(response.text)

        return {"result": data}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
