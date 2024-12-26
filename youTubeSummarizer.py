import os
import re
import gradio as gr
import torch
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    VideoUnavailable,
    NoTranscriptFound,
)
from youtube_transcript_api.formatters import TextFormatter

# summarization_pipe = pipeline(
#     task="summarization",
#     model="facebook/bart-large-cnn",
#     torch_dtype=torch.bfloat16,
#     device=0,
# )


model_path = "models/models--facebook--bart-large-cnn/snapshots/37f520fa929c961707657b28798b30c003dd100b"
summarization_pipe = pipeline(
    task="summarization", model=model_path, torch_dtype=torch.bfloat16
)


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

        youtube_summary = summarization_pipe(
            text_transcript, max_length=150, min_length=30, do_sample=False
        )
        return youtube_summary[0]["summary_text"]
    except TranscriptsDisabled:
        return "Transcript is disabled for this video."
    except VideoUnavailable:
        return "The video is unavailable."
    except NoTranscriptFound:
        return "No transcript found for this video."
    except Exception as e:
        return f"An error occurred: {e}"


grad_interface = gr.Interface(
    fn=get_youtube_transcript,
    inputs=gr.Textbox(label="Input your URL here: ", lines=1),
    outputs=gr.Textbox(label="Output summary is here: ", lines=10),
    title="Text Summarization Demo Project",
    description="This will summarize the video transcript",
)
grad_interface.launch(share=True)
