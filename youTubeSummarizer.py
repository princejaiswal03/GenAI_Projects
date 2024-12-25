import os
import re

import torch
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter

os.environ['HF_HOME'] = 'E:\Self\GenAI_Projects\models'

summarization_pipe = pipeline(task="summarization", model="facebook/bart-large-cnn", torch_dtype=torch.bfloat16)


# model_path = "models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"
# summarization_pipe = pipeline(task="summarization", model=model_path, torch_dtype=torch.bfloat16)


def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    """
    # Regex to extract the video ID from various YouTube URL formats
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, youtube_url)
    if match:
        return match.group(1)
    return None


def get_youtube_transcript(youtube_url):
    """
    Retrieves the transcript of a YouTube video using its URL.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return "Video Id can't be extracted from URL"

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Format the transcript into plain text
        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(transcript)

        youtube_summary = summarization_pipe(text_transcript, max_length=150, min_length=30, do_sample=False)
        return youtube_summary[0]['summary_text']
    except TranscriptsDisabled:
        return "Transcript is disabled for this video."
    except VideoUnavailable:
        return "The video is unavailable."
    except NoTranscriptFound:
        return "No transcript found for this video."
    except Exception as e:
        return f"An error occurred: {e}"


def main():
    youtube_url = input("Enter the YouTube video URL: ")
    transcript = get_youtube_transcript(youtube_url)

    if isinstance(transcript, list):
        # Print transcript in a readable format
        print("Transcript:")
        for entry in transcript:
            print(f"{entry['start']:.2f}s: {entry['text']}")
    else:
        # Print error message
        print(transcript)


if __name__ == "__main__":
    main()
