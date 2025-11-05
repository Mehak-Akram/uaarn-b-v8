import os
import json
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import RunConfig

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


router = APIRouter(prefix="/quiz", tags=["Quiz"])

class QuizRequest(BaseModel):
    topic: str

quiz_agent = Agent(
    name="quiz_agent",
    instructions="""
You are a Quiz Generator Agent.
- If user says "20 quiz", generate exactly 20 quiz questions.
- If user says "10 quiz", generate 10.
- If user says "3 quiz", generate 3.
- Default = 10 quizzes if user doesn't specify number.
Always respond in pure JSON format. 
add short explanations.


JSON FORMAT:
[
  {
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": "Correct option here"
  }
]

Rules:
- Only output JSON
- No markdown, no text before/after JSON
- Make sure JSON is valid
""",
)

def extract_json_from_text(text: str):
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except Exception:
            pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("No valid JSON found in AI output.")
@router.post("/")
async def generate_quiz(request: QuizRequest):
    if not request.topic or not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")

    prompt = f"Generate 5 quiz questions about {request.topic}."

    try:
        runner = Runner()
        result = await runner.run(
            starting_agent=quiz_agent,
            input=prompt,
            run_config=config
        )

        text_output = result.output_text if hasattr(result, "output_text") else str(result)

        parsed = extract_json_from_text(text_output)

        if not isinstance(parsed, list):
            raise ValueError("Parsed JSON is not a list")

        return {"quiz": parsed}

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"AI parse error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
