import torch
import gradio as gr
from diffusers import StableDiffusion3Pipeline

# Use a pipeline as a high-level helper
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "models/models--Salesforce--blip-image-captioning-large/snapshots/2227ac38c9f16105cb0412e7cab4759978a8fd90"

# caption_image = pipeline("image-to-text",
#                 model="Salesforce/blip-image-captioning-large", device=device)

caption_image = pipeline("image-to-text", model=model_path, device=device)


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


def caption_my_image(pil_image):
    semantics = caption_image(images=pil_image)[0]["generated_text"]
    image = image_generation(semantics)
    return image


demo = gr.Interface(
    fn=caption_my_image,
    inputs=[gr.Image(label="Select Image", type="pil")],
    outputs=[gr.Image(label="Generated Image", type="pil")],
    title="Project 8: Image Captioning",
    description="THIS APPLICATION WILL BE USED TO CAPTION THE IMAGE.",
)
demo.launch()
