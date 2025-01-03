import gradio as gr
from gradio_image_prompter import ImagePrompter
import json
from utils.drawing import draw_density_overlay
import io
import numpy as np
import pandas as pd
from PIL import Image
import requests

SERVER = "http://localhost:8000/model/predict"

def preprocess(points) -> str:
    if not points:
        return None
    boxes = pd.DataFrame(points)
    boxes = boxes[[1,0,4,3]].values.tolist()
    return json.dumps(boxes)


def predict(input):

    # Read Image (PIL) then convert to Binary
    image = Image.fromarray(input['image'].astype(np.uint8))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")  
    img_byte_arr.seek(0) 

    # process bounding boxes
    point = input['points']
    boxes = preprocess(point)

    num_shot = len(point) if len(point) <= 3 else 3

    files = {
        "image": ("img.png", img_byte_arr, "image/png") 
    }

    data = {
        "boxes": boxes
    }

    response = requests.post(SERVER, params={"num_shot": num_shot}, files=files, data=data)

    if response.status_code == 200:
        metadata = json.loads(response.headers.get("X-Metadata", "{}"))
        count_predict = metadata.get("count_predict", "Unknown")

        img_bytes = io.BytesIO(response.content)
        density_map = Image.open(img_bytes).convert("L")

        # Create visualizations
        overlay_image, density_map_img = draw_density_overlay(image, density_map, alpha=0.7)

        return overlay_image, density_map_img, round(count_predict)
    else:
        return None, None, None

    

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# Test app")
        with gr.Row():
            inp = ImagePrompter(label="Upload Image", show_label=False)

        with gr.Row():
            density_map = gr.Image(label="Density Map", show_label=True)
            density_image = gr.Image(label="Density Map on Image",show_label=True)

        textbox = gr.Textbox(label="Count prediction")
                       
        submit_button = gr.Button("Run")
        submit_button.click(fn=predict, inputs=inp, outputs=[density_image, density_map, textbox])
    demo.launch()
