from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
import eyed3
import wave
import logging
from pydub import AudioSegment
import math

# Load environment variables
load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_SIZE_MB = 25
OVERLAP_SECONDS = 1

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class AudioResponse(BaseModel):
    transcription: str
    summary: str
    tasks: str

def calculate_segment_duration_and_num_segments(duration_seconds, overlap_seconds, max_size, bitrate_kbps):
    """Calculate the duration and number of segments for an audio file."""
    seconds_for_max_size = (max_size * 8 * 1024) / bitrate_kbps
    num_segments = max(2, int(duration_seconds / seconds_for_max_size) + 1)
    total_overlap = (num_segments - 1) * overlap_seconds
    actual_playable_duration = (duration_seconds - total_overlap) / num_segments
    return num_segments, actual_playable_duration + overlap_seconds

def construct_file_names(path_to_audio, num_segments, extension):
    """Construct new file names for the segments of an audio file."""
    directory = os.path.dirname(path_to_audio)
    base_name = os.path.splitext(os.path.basename(path_to_audio))[0]
    padding = max(1, int(math.ceil(math.log10(num_segments))))
    new_names = [os.path.join(directory, f"{base_name}_{str(i).zfill(padding)}{extension}") for i in range(1, num_segments + 1)]
    return new_names

def split_audio(path_to_audio, overlap_seconds, max_size=MAX_SIZE_MB):
    """Split an audio file into segments."""
    if not os.path.exists(path_to_audio):
        raise ValueError(f"File {path_to_audio} does not exist.")
    
    extension = os.path.splitext(path_to_audio)[1].lower()

    # Handling mp3 files
    if extension == '.mp3':
        audio_file = eyed3.load(path_to_audio)
        if audio_file is None:
            raise ValueError(f"File {path_to_audio} is not a valid mp3 file.")
        duration_seconds = audio_file.info.time_secs
        bitrate_kbps = audio_file.info.bit_rate[1]

    # Handling wav files
    elif extension == '.wav':
        with wave.open(path_to_audio, 'rb') as wav_file:
            duration_seconds = wav_file.getnframes() / wav_file.getframerate()
            bitrate_kbps = wav_file.getsampwidth() * 8 * wav_file.getframerate() / 1000  # Calculating approximate bitrate

    else:
        raise ValueError(f"Unsupported file type: {extension}. Only .mp3 and .wav are supported.")

    file_size_MB = os.path.getsize(path_to_audio) / (1024 * 1024)
    if file_size_MB < max_size:
        logging.info("File is less than maximum size, no action taken.")
        return [path_to_audio]

    num_segments, segment_duration = calculate_segment_duration_and_num_segments(duration_seconds, overlap_seconds, max_size, bitrate_kbps)
    new_file_names = construct_file_names(path_to_audio, num_segments, extension)
    original_audio = AudioSegment.from_file(path_to_audio)
    start = 0
    for i in range(num_segments):
        if i == num_segments - 1:
            segment = original_audio[start:]
        else:
            end = start + segment_duration * 1000
            segment = original_audio[start:int(end)]
        segment.export(new_file_names[i], format=extension.lstrip('.'))
        start += (segment_duration - overlap_seconds) * 1000
    logging.info(f"Split into {num_segments} sub-files.")
    return new_file_names

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
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.mp3', '.wav']:
        return JSONResponse(status_code=400, content={"message": "Invalid file type. Only .mp3 and .wav are supported."})

    file_location = f"temp_{file.filename}"
    
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Split the file if it's too large
    segments = split_audio(file_location, OVERLAP_SECONDS)

    transcription = ""
    for segment in segments:
        transcription += transcribe_audio(segment)

    # Generate summary and tasks
    summary = generate_summary(transcription)
    tasks = generate_tasks(transcription)

    return AudioResponse(
        transcription=transcription,
        summary=summary,
        tasks=tasks
    )
