from diffusers import DiffusionPipeline
import torch
import streamlit as st


sdxl_base_model_path = ("../Models/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots"
                        "/462165984030d82259a11f4367a4eed129e94a7b")

color_book_lora_path = ".models/LoraColoringBook/ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"

color_book_lora_path ="artificialguybr/ColoringBookRedmond-V2"
color_book_trigger =", ColoringBookAF, Coloring Book"

@st.cache_resource
def load_pipeline(lora):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                                             torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                                             use_safetensors=True,
                                             variant="fp16" if device =="cuda" else None)

    # pipe = DiffusionPipeline.from_pretrained(sdxl_base_model_path,
    #                                          torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    #                                          use_safetensors=True,
    #                                          variant="fp16" if device == "cuda" else None)
    if lora != "None":
        pipe.load_lora_weights(color_book_lora_path)
    if device == "cuda":
        pipe.to(device)
    else:
        pipe.enable_model_cpu_offload()
    return  pipe

def image_generation(pipe, prompt, negative_prompt):
    try:
        image = pipe(
            prompt = prompt,
            negative_prompt = "blurred, ugly, watermark, low resolution" + negative_prompt,
            num_inference_steps= 20,
            guidance_scale=9.0
        ).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None


import streamlit as st

# Define the table as a list of dictionaries with the provided data
table = [
    {
        "name": "sai-neonpunk",
        "prompt": "neonpunk style . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    },
    {
        "name": "futuristic-retro cyberpunk",
        "prompt": "retro cyberpunk. 80's inspired, synthwave, neon, vibrant, detailed, retro futurism",
        "negative_prompt": "modern, desaturated, black and white, realism, low contrast"
    },
    {
        "name": "Dark Fantasy",
        "prompt": "Dark Fantasy Art, dark, moody, dark fantasy style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, bright, sunny"
    },
    {
        "name": "Double Exposure",
        "prompt": "Double Exposure Style, double image ghost effect, image combination, double exposure style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast"
    },
{
        "name": "None",
        "prompt": "8K ",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    }
]

# Convert the list of dictionaries to a dictionary with 'name' as key for easy lookup
styles_dict = {entry["name"]: entry for entry in table}




st.title("Project 12: @GenAILearniverse Coloring Book Generator")
prompt = st.text_input("Enter your Prompt", value="A cute Lion")

select_lora = st.selectbox("Select your lora", options=["Coloring Book", "None"])

# Dropdown for selecting a style
style_name = st.selectbox("Select a Style", options=list(styles_dict.keys()))

# Display the selected style's prompt and negative prompt
if style_name:
    selected_entry = styles_dict[style_name]
    selected_style_prompt = selected_entry["prompt"];
    selected_style_negative_prompt = selected_entry["negative_prompt"]
if st.button("Generate Awesome Image"):
    with st.spinner("Generating your awesome image..."):
        pipeline = load_pipeline(select_lora)
        if select_lora == "None":
            image =image_generation(pipeline,prompt + selected_style_prompt, selected_style_negative_prompt)
        else:
            image = image_generation(pipeline, prompt + selected_style_prompt + color_book_trigger, selected_style_negative_prompt)
        if image:
            st.image(image)