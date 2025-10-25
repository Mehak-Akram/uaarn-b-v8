from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from quiz import router as quiz_router  # ✅ Import your quiz router
from summarize import router as summarize_router
from ask import router as ask_router
from ask import app as ask_app

app = FastAPI()

# Optional: enable CORS if frontend will call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Include router
app.include_router(quiz_router, prefix="/quiz")
app.include_router(summarize_router, prefix="/summarize")
app.include_router(ask_router, prefix="/ask")


@app.get("/")
def root():
    return {"message": "🚀 UAARN Backend Running"}
