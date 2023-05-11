import gradio as gr
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import pinecone
import plotly.express as px
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.decomposition import PCA



def prints(s):
    print(f'[X] {s}')


def image_input(inp):
    # Prepare model input
    resized_image_rgb = inp.convert('RGB')
    prints('image converted')
    resized_image = resized_image_rgb.resize((224, 224))
    prints('image resized')
    input_nparr = np.array(resized_image)

    final_input = tf.keras.applications.resnet50.preprocess_input(input_nparr)
    prints('image preprocessed')
    final_input = final_input[None, :]

    # Instanciate model
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()

    # Feed input to model and predict
    print(final_input.shape)
    emb = avg_pool(model(final_input))
    prints('inference completed')

    # Query top 10 nearest embeddings
    print('[*] sending query')
    query_results = index.query(emb.numpy().tolist(), top_k=10)
    prints('query successful')

    # Save query results to global variable
    from pathlib import Path
    dataset_path = Path('full_dataset')
    results = []
    scores = []
    for r in query_results['matches']:
        filename = r['id'] + ".png"
        scores.append(r['score'])
        file_path = dataset_path / filename
        results.append(str(file_path))
    colorscale = px.colors.named_colorscales()[74]
    df = pd.DataFrame({
        "Results No.": list(range(1, 11)),
        "Scores": scores,
    })
    fig = px.bar(df, x="Results No.", y="Scores", color='Scores', color_continuous_scale=colorscale,
                 range_y=[min(scores), max(scores)],
                 )
    results = [(results[i], f"{i + 1}")
               for i in range(10)]
    return results, fig


def generate_viz():
    num_embeddings = 579
    vectors = index.fetch(ids=[str(i) for i in range(num_embeddings)])
    ids = []
    embeddings = []
    for _id, vector in vectors['vectors'].items():
        ids.append(_id)
        embeddings.append(vector['values'])
    embeddings = np.array(embeddings)
    pca2 = PCA(n_components=3).fit(embeddings)
    pca2d = pca2.transform(embeddings)
    import plotly.graph_objs as go
    scene = dict(xaxis=dict(title='PC1'), yaxis=dict(title='PC2'), zaxis=dict(title='PC3'))
    trace = go.Scatter3d(x=pca2d[:, 0], y=pca2d[:, 1], z=pca2d[:, 2], mode='markers')
    layout = go.Layout(margin=dict(l=0, r=0), scene=scene, height=1000, width=1000)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    return fig


with gr.Blocks(title='Big Data Project') as demo:
    with gr.Tabs():
        with gr.Tab('Image Search'):
            with gr.Column(variant="panel"):
                with gr.Row():
                    image = gr.Image(type="pil")
                with gr.Row():
                    btn_1 = gr.Button("Search image").style(full_width=True)
                with gr.Row():
                    gallery = gr.Gallery().style(columns=5, rows=2, object_fit="contain", height="auto")
                with gr.Row():
                    pl = gr.Plot()
                btn_1.click(image_input, image, outputs=[gallery, pl])
        with gr.Tab('Embeddings Visualization'):
            btn_2 = gr.Button("Generate").style(full_width=True)
            plot_3d = gr.Plot()
            btn_2.click(generate_viz, None, outputs=plot_3d)

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
