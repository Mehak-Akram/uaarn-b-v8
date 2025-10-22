import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file")

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend origin for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# External Gemini client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Run configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# Request body model
class QuizRequest(BaseModel):
    topic: str

# Create the quiz generation agent
agent = Agent(
    name="QuizAgent",
    instructions=(
        "You are a quiz generator AI. Generate multiple-choice questions for the given topic. "
        "Each question must have 4 options and one correct answer. "
        "Respond ONLY in valid JSON format like this:\n\n"
        "["
        "{\"question\": \"What is Python?\", \"options\": [\"Snake\", \"Language\", \"Game\", \"OS\"], \"answer\": \"Language\"},"
        "{\"question\": \"2+2?\", \"options\": [\"1\", \"2\", \"3\", \"4\"], \"answer\": \"4\"}"
        "]"
    ),
)

# API endpoint
@app.post("/quiz")
async def generate_quiz(req: QuizRequest):
    """
    Generate a multiple-choice quiz using Gemini model through OpenAI Agents SDK.
    """
    try:
        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required")

        result = await Runner.run(agent, f"Generate quiz on {topic}", run_config=config)

        # Get the output from the AI agent
        output = getattr(result, "output_text", None) or getattr(result, "output", None)
        if not output:
            raise HTTPException(status_code=500, detail="No output returned from AI")

        # Ensure it's valid JSON
        try:
            json.loads(output)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid JSON format from AI")

        return {"quiz": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}")

# Root route (optional)
@app.get("/")
def root():
    return {"message": "✅ AI Quiz Backend is running!"}
