from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set your OpenAI API key

app = FastAPI()




# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class AudioResponse(BaseModel):
    transcription: str
    summary: str
    tasks: str

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcription = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcription

def generate_summary(transcript):
    prompt = (
        "Summarize the following transcript:\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Summary:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant who summarizes spoken text."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

def generate_tasks(transcript):
    prompt = (
        "From the following transcript, generate a list of tasks and to-dos:\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Tasks:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant who extracts tasks from spoken text."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

@app.post("/process-audio/", response_model=AudioResponse)
async def process_audio(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Transcribe the uploaded audio
    transcription = transcribe_audio(file_location)

    # Generate summary and tasks
    summary = generate_summary(transcription)
    tasks = generate_tasks(transcription)

    return AudioResponse(
        transcription=transcription,
        summary=summary,
        tasks=tasks
    )

