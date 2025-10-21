import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

class QuizRequest(BaseModel):
    topic: str


agent = Agent(
    name="QuizAgent",
    instructions=(
        "You are a quiz generator AI. Generate  multiple-choice "
        "questions for the given topic. Each question must have 4 options "
        "and one correct answer. Respond ONLY in valid JSON format like this:\n\n"
        "["
        "{\"question\": \"What is Python?\", \"options\": [\"Snake\", \"Language\", \"Game\", \"OS\"], \"answer\": \"Language\"},"
        "{\"question\": \"2+2?\", \"options\": [\"1\", \"2\", \"3\", \"4\"], \"answer\": \"4\"}"
        "]"
    ),
)


@app.post("/quiz")
async def generate_quiz(req: QuizRequest):
    try:
        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required")

        result = await Runner.run(agent, f"Generate quiz on {topic}", run_config=config)

        output = getattr(result, "output_text", None) or getattr(result, "output", None)
        if not output:
            raise HTTPException(status_code=500, detail="No output returned from AI")

        return {"quiz": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}")
