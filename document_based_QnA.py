import torch
import gradio as gr
import json

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = "models/models--deepset--roberta-base-squad2/snapshots/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"

# question_answer = pipeline("question-answering",
#                            model="deepset/roberta-base-squad2")


question_answer = pipeline("question-answering", model=model_path)


def read_file_content(file_obj):
    """
    Reads the content of a file object and returns it.
    Parameters:
    file_obj (file object): The file object to read from.
    Returns:
    str: The content of the file.
    """
    try:
        with open(file_obj.name, "r", encoding="utf-8") as file:
            context = file.read()
            return context
    except Exception as e:
        return f"An error occurred: {e}"


# Example usage:
# with open('example.txt', 'r') as file:
#     content = read_file_content(file)
#     print(content)


# context =("Mark Elliot Zuckerberg (/ˈzʌkərbɜːrɡ/; born May 14, 1984) is an American businessman. He co-founded the social media service Facebook, along with his Harvard roommates in 2004, and its parent company Meta Platforms (formerly Facebook, Inc.), of which he is chairman, chief executive officer and controlling shareholder.Zuckerberg briefly attended Harvard University, where he launched Facebook "
#           "in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, "
#           "Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with "
#           "majority shares. In 2008, at age 23, he became the world's youngest self-made billionaire. "
#           "He has since used his funds to organize multiple donations, including the establishment "
#           "of the Chan Zuckerberg Initiative.")
# question ="what is Mark's DOB?"


def get_answer(file, question):
    context = read_file_content(file)
    answer = question_answer(question=question, context=context)
    return answer["answer"]


demo = gr.Interface(
    fn=get_answer,
    inputs=[
        gr.File(label="Upload your file"),
        gr.Textbox(label="Input your question", lines=1),
    ],
    outputs=[gr.Textbox(label="Answer text", lines=1)],
    title="Project 5: Document Q & A",
    description="THIS APPLICATION WILL BE USED TO ANSER QUESTIONS BASED ON CONTEXT PROVIDED.",
)

demo.launch()
