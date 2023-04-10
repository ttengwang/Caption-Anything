from io import BytesIO
import string
import gradio as gr
import requests
from caas import CaptionAnything
import torch
import json
import sys
import argparse
from caas import parse_augment
import numpy as np
import PIL.ImageDraw as ImageDraw

title = """<h1 align="center">Caption-Anything</h1>"""
description = """Gradio demo for Caption Anything, image to dense captioning generation with various language styles. To use it, simply upload your image, or click one of the examples to load them.
<br> <strong>Code</strong>: GitHub repo: < a href=' ' target='_blank'></ a>
"""

examples = [
    ["test_img/img2.jpg", "[[1000, 700, 1]]"]
]

args = parse_augment()
args.device = 'cuda:1'
args.disable_gpt = True
args.enable_reduce_tokens = True
args.port=20321

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
    
def inference_seg_cap(image_input, chat_input, language, sentiment, factuality, length, state, click_state, evt:gr.SelectData):
    controls = {'length': length,
             'sentiment': sentiment,
             'factuality': factuality,
             'language': language}
    click_coordinate = "[[{}, {}, 1]]".format(str(evt.index[0]), str(evt.index[1])) 
    chat_input = click_coordinate
    prompt = get_prompt(chat_input, click_state)
    print('prompt: ', prompt, 'controls: ', controls)
    out = model.inference(image_input, prompt, controls)
    state = state + [(None, "Image point: {}, Input label: {}".format(prompt["input_point"], prompt["input_label"]))]
    for k, v in out['generated_captions'].items():
        state = state + [(f'{k}: {v}', None)]
    
    text = out['generated_captions']
    draw = ImageDraw.Draw(image_input)
    draw.text((evt.index[0], evt.index[1]), text, textcolor=(0,0,255), text_size=120) 
    return text, click_state, chat_input, image_input


def upload_callback(image_input, state):
    state = state + [('Image size: ' + str(image_input.size), None)]
    return state


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
        with gr.Column(scale=0.7):
            image_input = gr.Image(type="pil", interactive=True, onchange=upload_callback).style(height=680,width=1280, scale=1.0)

            language = gr.Radio(
                choices=["English", "Chinese", "French", "Spanish", "Arabic", "Portuguese","Cantonese"],
                value="English",
                label="Language",
                interactive=True,
            )
            sentiment = gr.Radio(
                choices=["Positive", "Negative"],
                value="Positive",
                label="Sentiment",
                interactive=True,
            )
            factuality = gr.Radio(
                choices=["Factual", "Imagination"],
                value="Factual",
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

    with gr.Row(scale=1.5):
        with gr.Row():
            image_output_mask= gr.Image(type="pil", interactive=False).style(height=260,scale=1.0)
            image_output_crop= gr.Image(type="pil", interactive=False).style(height=260,scale=1.0)
        # chatbot = gr.Chatbot(label="Chat Output",).style(height=500,scale=0.5)
        with gr.Column():
        # with gr.Column(scale=1):
            chat_input = gr.Textbox(lines=1, label="Chat Input")
            # chat_input.submit(
            #     inference_seg_cap,
            #     [
            #         image_input,
            #         chat_input,
            #         language,
            #         sentiment,
            #         factuality,
            #         length,
            #         state,
            #         click_state
            #     ],
            #     [state, click_state, image_output_mask, image_output_crop],
            # )
            
            image_input.upload(
                upload_callback,
                [image_input, state],
                [state]
                )

            with gr.Row():
                clear_button = gr.Button(value="Clear Click", interactive=True)
                clear_button.click(
                    lambda: ("", [[], []], [], []),
                    [],
                    [chat_input, click_state, image_output_mask, image_output_crop],
                    queue=False,
                )
                
                clear_button = gr.Button(value="Clear", interactive=True)
                clear_button.click(
                    lambda: ("", [], [], [[], []], [], []),
                    [],
                    [chat_input, state, click_state, image_output_mask, image_output_crop],
                    queue=False,
                )

                # submit_button = gr.Button(
                #     value="Submit", interactive=True, variant="primary"
                # )
                # submit_button.click(
                #     inference_seg_cap,
                #     [
                #         image_input,
                #         chat_input,
                #         language,
                #         sentiment,
                #         factuality,
                #         length,
                #         state,
                #         click_state
                #     ],
                #     [state, click_state, image_output_mask, image_output_crop],
                # )

        # select coordinate
        image_input.select(inference_seg_cap, 
                           inputs=[
                                    image_input,
                                    chat_input,
                                    language,
                                    sentiment,
                                    factuality,
                                    length,
                                    state,
                                    click_state
                                    ],
                            outputs=[state, click_state, chat_input, image_input])
    

        image_input.change(
            lambda: ([], [[], []]),
            [],
            [state, click_state],
            queue=False,
        )
    examples = gr.Examples(
        examples=examples,
        inputs=[image_input, chat_input],
    )

iface.queue(concurrency_count=1, api_open=False, max_size=10)
iface.launch(server_name="0.0.0.0", enable_queue=True, server_port=args.port, share=args.gradio_share)