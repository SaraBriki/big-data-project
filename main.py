import gradio as gr
import random

results = ['' for i in range(3)]


def image_input(inp):
    gal = ["images/karsten-winegeart-oU6KZTXhuvk-unsplash.jpg",
           "images/karsten-winegeart-Qb7D1xw28Co-unsplash.jpg",
           "images/sq-lim-k4vhuUHv08o-unsplash.jpg",
           ]
    for i in range(3):
        results[i] = gal[i]
    print("yay user input is here!")
    return results


def generate_gallery():
    images = [
        (random.choice(results), f"Image result {i + 1}")
        for i in range(3)
    ]
    return images


with gr.Blocks() as demo:
    with gr.Column(variant="panel"):
        with gr.Row():
            image = gr.Image()
            image.change(fn=image_input, inputs=image, outputs=None)
        with gr.Row():
            btn = gr.Button("Search image").style(full_width=True)
        with gr.Row():
            gallery = gr.Gallery(
                label="Result images", show_label=False, elem_id="gallery"
            ).style(columns=3, rows=1, object_fit="contain", height="auto")
            btn.click(generate_gallery, None, gallery)
demo.launch()
