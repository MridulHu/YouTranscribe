from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")


genai.configure(api_key=API_KEY)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoIdInput(BaseModel):
    video_id: str


@app.get("/api/transcript/{video_id}")
def get_transcript(video_id: str):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return {"transcript": transcript}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/generate")
async def summarize_transcript(data: VideoIdInput):
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(data.video_id)
        full_text = " ".join([item["text"] for item in transcript_data])

        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
        prompt = f"Summarize the following YouTube transcript:\n\n{full_text}"
        response = model.generate_content(prompt)

        print("=== Gemini response ===")
        print(response.text) 
        print("=======================")

        return {"summary": response.text}
    except Exception as e:
        print("Error during summarization:", str(e))
        return {"error": str(e)}

