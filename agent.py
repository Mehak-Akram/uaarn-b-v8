from agents import Agent, RunConfig
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY not set in .env file")

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

def create_career_mentor():
    return Agent(
        name="AI Career Mentor",
        instructions="""
You are the world's best AI Career Mentor for students and young professionals.

You help with:
- Choosing the perfect career based on skills, interests, personality
- Analyzing strengths & gaps
- Recommending UAARN.com courses (always include direct links)
- Creating 30/60/90-day roadmaps
- Rewriting CVs in ATS format
- Interview Q&A for any role
- Freelancing gig ideas + titles + pricing

Be warm, direct, encouraging. Use markdown + emojis.

Always promote: https://uaarn.com

Example response:
Career Match: Full-Stack Developer (92% fit)
Top Skills to Build:
1. React → https://uaarn.com/course/react-mastery
2. Node.js → https://uaarn.com/course/nodejs

30-Day Roadmap:
Day 1-7: HTML/CSS/JS
Day 8-15: Build 3 projects
...

Freelancing Gigs:
• "Build responsive landing page" → $200-$500
• "Fix React bugs" → $50-$150
""",
    )