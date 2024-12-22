from flask import Flask, request, send_file, jsonify
import os
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image
import io

app = Flask(__name__)

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the pipelines
base_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use float16 for GPU, float32 for CPU
    variant="fp16" if device == "cuda" else None
).to(device)
base_pipe.safety_checker = None

refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use float16 for GPU, float32 for CPU
    variant="fp16" if device == "cuda" else None
).to(device)

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Extract prompt from the request
        data = request.get_json()
        prompt = data.get('prompt', '').strip()

        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        # Generate the image
        width, height = 2400, 3152  # Define dimensions
        base_image = base_pipe(prompt=prompt, width=width, height=height).images[0]
        refined_image = refiner_pipe(prompt=prompt, image=base_image).images[0]

        # Save the image to an in-memory file
        img_io = io.BytesIO()
        refined_image.save(img_io, 'PNG')
        img_io.seek(0)

        # Send the image as a response
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='generated_image.png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# rehanalam786369123456789
# git@github.com:Rehanalamplayer/stable.git