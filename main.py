from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from quiz import router as quiz_router
from summarize import router as summarize_router
from ask import router as ask_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(quiz_router)
app.include_router(summarize_router)
app.include_router(ask_router)

@app.get("/")
def root():
    return {"message": "🚀 UAARN Backend Running"}
