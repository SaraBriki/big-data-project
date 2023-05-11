import gradio as gr
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import pinecone
import plotly.express as px
from dotenv import load_dotenv


# results = ['' for i in range(3)]

def prints(s):
    print(f'[X] {s}')


def image_input(inp):
    # Prepare model inpu
    resized_image_rgb = inp.convert('RGB')
    prints('image converted')
    resized_image = resized_image_rgb.resize((224, 224))
    prints('image resized')
    input_nparr = np.array(resized_image)

    # model_input = tf.reshape(input_nparr, [1, 224, 224, 3])
    final_input = tf.keras.applications.resnet50.preprocess_input(input_nparr)
    prints('image preprocessed')
    final_input = final_input[None, :]

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
    results = []
    for r in query_results['matches']:
        filename = r['id'] + ".png"
        file_path = dataset_path / filename
        results.append(file_path)
        print(file_path)
    colorscale = px.colors.named_colorscales()[3]
    df = pd.DataFrame({
        "Fruits": ["Apples", "Oranges", "Bananas", "Pineapple", "Strawberry", "Mango", "Blueberry", "Tomatoes", "Watermelon", "Melon", "Peach"],
        "Scores": [4, 3.5, 8, 5, 6, 6, 3, 5, 7, 6, 10],
    })
    fig = px.bar(df, x="Fruits", y="Scores", color='Scores', color_continuous_scale=colorscale,
                 )
    results.append(fig)
    return results


with gr.Blocks() as demo:
    with gr.Column(variant="panel"):
        with gr.Row():
            image = gr.Image(type="pil")
        with gr.Row():
            btn = gr.Button("Search image").style(full_width=True)
        with gr.Row():
            out1 = gr.Image(type="filepath")
            out2 = gr.Image(type="filepath")
            out3 = gr.Image(type="filepath")
        with gr.Row():
            out4 = gr.Plot()
            btn.click(image_input, image, outputs=[out1, out2, out3, out4])

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
