from io import BytesIO
import string
import gradio as gr
import requests
from caas import CaptionAnything
import torch
import json
import sys
import argparse

title = """<h1 align="center">Caption-Anything</h1>"""
description = """Gradio demo for Caption Anything, image to dense captioning generation with various language styles. To use it, simply upload your image, or click one of the examples to load them.
<br> <strong>Code</strong>: GitHub repo: <a href='https://github.com/ttengwang/Caption-Anything' target='_blank'></a>
"""

examples = [
    ["test_img/img2.jpg", "hi"]
]

parser = argparse.ArgumentParser()
parser.add_argument('--captioner', type=str, default="blip")
parser.add_argument('--segmenter', type=str, default="base")
parser.add_argument('--text_refiner', type=str, default="base")
parser.add_argument('--segmenter_checkpoint', type=str, default="segmenter/sam_vit_h_4b8939.pth")
parser.add_argument('--seg_crop_mode', type=str, default="w_bg", help="whether to add or remove background of the image when captioning")
parser.add_argument('--clip_filter', action="store_true", help="use clip to filter bad captions")
parser.add_argument('--device', type=str, default="cuda:0")    
parser.add_argument('--port', type=int, default=6086)    
args = parser.parse_args()

# args = type('args', (object,), {})()
# args.captioner='blip'
# args.segmenter='base'
# args.text_refiner = 'base'
# args.segmenter_checkpoint = 'segmenter/sam_vit_h_4b8939.pth'
# args.seg_crop_mode = 'w_bg'
# args.clip_filter = False
# args.device = "cuda" if torch.cuda.is_available() else "cpu"

model = CaptionAnything(args)

def get_prompt(chat_input, click_state):    
    points = click_state[0]
    labels = click_state[1]
    inputs = json.loads(chat_input)
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    
    prompt = {
        "prompt_type":["click"],
        "input_point":points,
        "input_label":labels,
        "multimask_output":"True",
    }
    return prompt
    
def inference_seg_cap(image_input, chat_input, language, sentiment, factuality, length, state, click_state):
    controls = {'length': length,
             'sentiment': sentiment,
             'factuality': factuality,
             'language': language}
    prompt = get_prompt(chat_input, click_state)
    print('prompt: ', prompt, 'controls: ', controls)
    text_dict = model.inference(image_input, prompt, controls)
    chatbot = state + [(text_dict['raw_caption'], None), (text_dict['caption'], None), (text_dict['caption'], None)]
    return chatbot, chatbot, click_state

    
with gr.Blocks(
    css="""
    .message.svelte-w6rprc.svelte-w6rprc.svelte-w6rprc {font-size: 20px; margin-top: 20px}
    #component-21 > div.wrap.svelte-w6rprc {height: 600px;}
    """
) as iface:
    state = gr.State([])
    click_state = gr.State([[],[]])

    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", interactive=True).style(height=260,scale=1.0)

            # with gr.Row():
            language = gr.Radio(
                choices=["English", "Chinese", "French", "Spanish", "Arabic", "Portuguese","Cantonese"],
                value="Language",
                label="Language",
                interactive=True,
            )
            sentiment = gr.Radio(
                choices=["Positive", "Negative"],
                value="Sentiment",
                label="Sentiment",
                interactive=True,
            )
            factuality = gr.Radio(
                choices=["Factual", "Imagination"],
                value="Factuality",
                label="Factuality",
                interactive=True,
            )
            length = gr.Slider(
                minimum=5,
                maximum=100,
                value=10,
                step=1,
                interactive=True,
                label="Length",
            )

        with gr.Column(scale=1.5):
            with gr.Row():
                chatbot = gr.Chatbot(label="Chat Output",).style(height=500,scale=0.5)
            with gr.Row():
            # with gr.Column(scale=1):
                chat_input = gr.Textbox(lines=1, label="Chat Input")
                chat_input.submit(
                    inference_seg_cap,
                    [
                        image_input,
                        chat_input,
                        language,
                        sentiment,
                        factuality,
                        length,
                        state,
                        click_state
                    ],
                    [chatbot, state, click_state],
                )

                with gr.Row():
                    clear_button = gr.Button(value="Clear Click", interactive=True)
                    clear_button.click(
                        lambda: ("", [], [], [[], []]),
                        [],
                        [chat_input, chatbot, state, click_state],
                        queue=False,
                    )
                    
                    clear_button = gr.Button(value="Clear", interactive=True)
                    clear_button.click(
                        lambda: ("", [], [], [[], []]),
                        [],
                        [chat_input, chatbot, state, click_state],
                        queue=False,
                    )

                    submit_button = gr.Button(
                        value="Submit", interactive=True, variant="primary"
                    )
                    submit_button.click(
                        inference_seg_cap,
                        [
                            image_input,
                            chat_input,
                            language,
                            sentiment,
                            factuality,
                            length,
                            state,
                            click_state
                        ],
                        [chatbot, state],
                    )

            image_input.change(
                lambda: ("", [], [[], []]),
                [],
                [chatbot, state, click_state],
                queue=False,
            )

    examples = gr.Examples(
        examples=examples,
        inputs=[image_input, chat_input],
    )

iface.queue(concurrency_count=1, api_open=False, max_size=10)
iface.launch(server_name="0.0.0.0", enable_queue=True, server_port=args.port)
