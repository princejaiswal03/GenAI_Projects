from flask import Flask, render_template, request
from diffusers import StableDiffusion3Pipeline
import torch
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the text-to-image model
model_id = "../models/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)


# Route for the home page
@app.route("/", methods=["GET", "POST"])
def home():
    generated_image_path = None

    if request.method == "POST":
        prompt = request.form.get("prompt")
        if prompt:
            # Generate image from prompt
            image = pipe(prompt).images[0]
            # Save the image locally
            generated_image_path = os.path.join("static", "generated_image.png")
            image.save(generated_image_path)

    return render_template("index.html", image_path=generated_image_path)


if __name__ == "__main__":
    app.run(debug=True)
