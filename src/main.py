import gradio as gr
import os
import shutil
import time
import pandas as pd
import numpy as np
import uuid
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import pinecone
from dotenv import load_dotenv


# results = ['' for i in range(3)]

def prints(s):
    print(f'[X] {s}')
def image_input(inp):

    # Prepare model input

    resized_image_rgb = inp.convert('RGB')
    prints('image converted')
    resized_image = resized_image_rgb.resize((224, 224))
    prints('image resized')
    input_nparr = np.array(resized_image)
    # model_input = tf.reshape(input_nparr, [1, 224, 224, 3])
    final_input = tf.keras.applications.resnet50.preprocess_input(input_nparr)
    prints('image preprocessed')
    final_input = final_input[None,: ]

    # Instanciate model

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()

    # Feed input to model and predict
    print(final_input.shape)
    emb = avg_pool(model(final_input))
    prints('inference completed')
    # Query top 3 nearest embeddings
    print('[*] sending query')
    query_results = index.query(emb.numpy().tolist(), top_k=3)
    prints('query successful')
    # Save query results to global variable
    from pathlib import Path
    dataset_path = Path('full_dataset')
    results=[]
    for r in query_results['matches']:
        filename = r['id']+".png"
        file_path = dataset_path/filename
        results.append(file_path)
        print(file_path)
    print("yay user input is here!")
    return results


# def generate_gallery():
#     images = [
#         (str(results[i]), f"Image result {i + 1}")
#         for i in range(3)
#     ]
#     print(images)
#     return images


with gr.Blocks() as demo:
    with gr.Column(variant="panel"):
        with gr.Row():
            image = gr.Image(type="pil")
            # image.change(fn=image_input, inputs=image, outputs=None)
        with gr.Row():
            btn = gr.Button("Search image").style(full_width=True)
        with gr.Row():
            # gallery = gr.Gallery(
            #     label="Result images", show_label=False, elem_id="gallery"
            # ).style(columns=3, rows=1, object_fit="contain", height="auto")
            out1 = gr.Image(type="filepath")
            out2 = gr.Image(type="filepath")
            out3 = gr.Image(type="filepath")
            btn.click(image_input, image, outputs=[out1,out2,out3])

# Connect to pinecone environment
load_dotenv()
prints('dotenv loaded')
pinecone.init(
    api_key=os.environ.get('API_KEY'),
    environment="us-east-1-aws"  # find next to API key in console
)
index_name = "image-search"
index = pinecone.Index(index_name)
prints('pinecone initialized')
model = ResNet50(include_top=False)
prints('model loaded')
demo.launch()
