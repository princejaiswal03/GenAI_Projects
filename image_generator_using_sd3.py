import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline


def image_generation(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "models/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        text_encoder_3=None,
        tokenizer_3=None,
    )
    if device == "cuda":
        pipeline.to(device)
    else:
        pipeline.enable_model_cpu_offload()

    image = pipeline(
        prompt=prompt,
        negative_prompt="blurred, ugly, watermark, low resolution, blurry",
        num_inference_steps=40,
        height=1024,
        width=1024,
        guidance_scale=9.0,
    ).images[0]

    return image


# image_generation("A magician cat doing spell")

interface = gr.Interface(
    fn=image_generation,
    inputs=gr.Textbox(lines=2, placeholder="Enter your Prompt..."),
    outputs=gr.Image(type="pil"),
    title="@GenAiLearnivers Project 9: Image creation using Stable Diffusion 3 Model",
    description="This application will be used to generate awesome images using SD3 model",
)

interface.launch()
