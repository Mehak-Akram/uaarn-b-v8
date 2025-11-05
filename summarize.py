import os
import io
from fastapi import APIRouter, FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv
from langdetect import detect
from gtts import gTTS
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import RunConfig

router = APIRouter(prefix="/summarize", tags=["Summarize"])



load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("⚠ GEMINI_API_KEY not set in .env file")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    source: str
    link: str | None = None
    text: str | None = None

class TTSRequest(BaseModel):
    text: str


def create_agent():
    return Agent(
        name="Summarizer Agent",
        instructions="""
        You are a smart summarization assistant.
        Summarize clearly and concisely.
        If YouTube link is provided, summarize content.
        If raw text or transcript is provided, give structured summary.
        Return result in markdown format:
        📌 Short Summary: ...
        📝 Key Points:
        - Point 1
        - Point 2
        ⏳ Timestamps or Highlights: ...
        🧠 Description: ...
        """
    )

async def translate_to_english(text: str) -> str:
    try:
        detected_lang = detect(text)
        if detected_lang.lower() == "en":
            return text
        translation_prompt = f"Translate this text from {detected_lang} to English:\n\n{text[:40000]}"
        translation_agent = Agent(name="Translation Agent", instructions="Translate text accurately to English.")
        result = await Runner.run(translation_agent, translation_prompt, run_config=config)
        return result.final_output
    except Exception as e:
        print(f"⚠ Translation error: {e}")
        return text

@router.post("/api/agent/summarize")
async def summarize(req: SummarizeRequest):
    agent = create_agent()

    if req.source == "youtube" and req.link:
        user_prompt = f"Summarize this YouTube video:\n{req.link}"
    elif req.source == "text" and req.text:
        translated_text = await translate_to_english(req.text)
        user_prompt = f"Summarize the following transcript:\n{translated_text[:40000]}"
    else:
        raise HTTPException(status_code=400, detail="Missing input")

    result = await Runner.run(agent, user_prompt, run_config=config)
    return {"output": result.final_output}

@router.post("/api/agent/upload")
async def upload_file(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8", errors="ignore")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    translated_text = await translate_to_english(content)
    agent = create_agent()
    user_prompt = f"Summarize the following transcript from file:\n{translated_text[:40000]}"
    result = await Runner.run(agent, user_prompt, run_config=config)
    return {"output": result.final_output}

@router.post("/api/agent/tts")
async def text_to_speech(req: TTSRequest):
    try:
        tts = gTTS(text=req.text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return StreamingResponse(mp3_fp, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


@router.post("/api/agent/download/txt")
async def download_txt(req: TTSRequest):
    text_bytes = io.BytesIO(req.text.encode('utf-8'))
    return StreamingResponse(
        text_bytes,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=summary.txt"}
    )


@router.post("/api/agent/download/pdf")
async def download_pdf(req: TTSRequest):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    textobject = p.beginText(40, 750)
    for line in req.text.splitlines():
        textobject.textLine(line)
    p.drawText(textobject)
    p.showPage()
    p.save()
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=summary.pdf"}
)

