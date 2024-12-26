import os

import gradio as gr
import torch
from transformers import pipeline

# os.environ['HF_HOME'] = 'E:\Self\GenAI_Projects\models'

summarization_pipe = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    torch_dtype=torch.bfloat16,
    device=0,
)


# model_path = "models/hub/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"
# summarization_pipe = pipeline(task="summarization", model=model_path, torch_dtype=torch.bfloat16)

# text_input = '''
# Five soldiers were killed on Tuesday (December 24, 2024) evening when a vehicle skidded off the road during operational duty in the Pir Panjal valley’s Poonch district, the Army said.
# A spokesman of the Army’s White Knight Corps, in charge of the Rajouri-Poonch belt, said, “Five brave soldiers lost their lives in a vehicle accident during operational duty in the Poonch sector”.
# Rescue operations were going on in the area, the Army said. “Injured personnel are receiving medical care,” the Army added.
# Preliminary reports suggested at least seven soldiers was onboard of the vehicle when it met with an accident and fell into a gorge Balnoi Ghora area. The vehicle was travelling from Nilam headquarters to a post at Balnoi Ghora.
# '''
# print(pipe(text_input))


def summarize_text(input_text):
    output = summarization_pipe(input_text)
    return output[0]["summary_text"]


# grad_interface = gr.Interface(fn=summarize_text, inputs="text", outputs="text")
grad_interface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(label="Input your text here: ", lines=10),
    outputs=gr.Textbox(label="Output text is here: ", lines=10),
    title="Text Summarization Demo Project",
    description="This will summarize the text input",
)
grad_interface.launch(share=True)
