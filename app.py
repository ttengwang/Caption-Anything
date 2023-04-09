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

title = """<h1 align="center">Caption-Anything</h1>"""
description = """Gradio demo for Caption Anything, image to dense captioning generation with various language styles. To use it, simply upload your image, or click one of the examples to load them.
<br> <strong>Code</strong>: GitHub repo: <a href='https://github.com/ttengwang/Caption-Anything' target='_blank'></a>
"""

examples = [
    ["test_img/img2.jpg", "[[1000, 700, 1]]"]
]

args = parse_augment()

# args = type('args', (object,), {})()
# args.captioner='blip'
# args.segmenter='base'
# args.text_refiner = 'base'
# args.segmenter_checkpoint = 'segmenter/sam_vit_h_4b8939.pth'
# args.seg_crop_mode = 'w_bg'
# args.clip_filter = False
# args.device = "cuda" if torch.cuda.is_available() else "cpu"

model = CaptionAnything(args)
# model = None

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
    out = model.inference(image_input, prompt, controls)
    state = state + [(None, "Image point: {}, Input label: {}".format(prompt["input_point"], prompt["input_label"]))]
    for k, v in out['generated_captions'].items():
        state = state + [(f'{k}: {v}', None)]
    image_output_mask = out['mask_save_path']
    image_output_crop = out['crop_save_path']
    return state, state, click_state, image_output_mask, image_output_crop


def upload_callback(image_input, state):
    state = state + [('Image size: ' + str(image_input.size), None)]
    return state

# get coordinate in format [[x,y,positive/negative]]
def get_select_coords(image_input, point_prompt, language, sentiment, factuality, length, state, click_state, evt: gr.SelectData):
        print("point_prompt: ", point_prompt)
        if point_prompt == 'Positive Point':
            coordinate = "[[{}, {}, 1]]".format(str(evt.index[0]), str(evt.index[1]))
        else:
            coordinate = "[[{}, {}, 0]]".format(str(evt.index[0]), str(evt.index[1]))
            
        return (coordinate,) + inference_seg_cap(image_input, coordinate, language, sentiment, factuality, length, state, click_state)


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
            image_input = gr.Image(type="pil", interactive=True, label="Image").style(height=260,scale=1.0)

            with gr.Row(scale=0.7):
                point_prompt = gr.Radio(
                    choices=["Positive Point",  "Negative Point"],
                    value="Positive",
                    label="Points",
                    interactive=True,
                )
            
            # with gr.Row():
            language = gr.Radio(
                choices=["English", "Chinese", "French", "Spanish", "Arabic", "Portuguese","Cantonese"],
                value="English",
                label="Language",
                interactive=True,
            )
            sentiment = gr.Radio(
                choices=["Positive", "Natural", "Negative"],
                value="Natural",
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

        with gr.Column(scale=1.5):
            with gr.Row():
                image_output_mask= gr.Image(type="pil", interactive=False, label="Mask").style(height=260,scale=1.0)
                image_output_crop= gr.Image(type="pil", interactive=False, label="Cropped Image by Mask").style(height=260,scale=1.0)
            chatbot = gr.Chatbot(label="Chat Output",).style(height=500,scale=0.5)
            with gr.Row():
            # with gr.Column(scale=1):
                chat_input = gr.Textbox(lines=1, label="Input Prompt (A list of points lke : [[100, 200, 1], [200,300,0]])")
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
                    [chatbot, state, click_state, image_output_mask, image_output_crop],
                )
                
                image_input.upload(
                    upload_callback,
                    [image_input, state],
                    [chatbot]
                    )

                with gr.Row():
                    clear_button = gr.Button(value="Clear Click", interactive=True)
                    clear_button.click(
                        lambda: ("", [[], []], None, None),
                        [],
                        [chat_input, click_state, image_output_mask, image_output_crop],
                        queue=False,
                    )
                    
                    clear_button = gr.Button(value="Clear", interactive=True)
                    clear_button.click(
                        lambda: ("", [], [], [[], []], None, None),
                        [],
                        [chat_input, chatbot, state, click_state, image_output_mask, image_output_crop],
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
                        [chatbot, state, click_state, image_output_mask, image_output_crop],
                    )

            # select coordinate
            image_input.select(
                get_select_coords, 
                inputs=[image_input,point_prompt,language,sentiment,factuality,length,state,click_state], 
                outputs=[chat_input, chatbot, state, click_state, image_output_mask, image_output_crop]
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
iface.launch(server_name="0.0.0.0", enable_queue=True, server_port=args.port, share=args.gradio_share)
