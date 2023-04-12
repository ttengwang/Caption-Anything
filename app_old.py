from io import BytesIO
import string
import gradio as gr
import requests
from caption_anything import CaptionAnything
import torch
import json
import sys
import argparse
from caption_anything import parse_augment
import os

# download sam checkpoint if not downloaded
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return filepath
checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
folder = "segmenter"
filename = "sam_vit_h_4b8939.pth"

title = """<h1 align="center">Caption-Anything</h1>"""
description = """Gradio demo for Caption Anything, image to dense captioning generation with various language styles. To use it, simply upload your image, or click one of the examples to load them.
<br> <strong>Code</strong>: GitHub repo: <a href='https://github.com/ttengwang/Caption-Anything' target='_blank'></a>
"""

examples = [
    ["test_img/img2.jpg", "[[1000, 700, 1]]"]
]

args = parse_augment()

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
    click_state[2].append(out['generated_captions']['raw_caption'])
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
    
def chat_with_points(chat_input, click_state, state):
    points, labels, captions = click_state
    # point_chat_prompt = "I want you act as a chat bot in terms of image. I will give you some points (w, h) in the image and tell you what happed on the point in natural language. Note that (0, 0) refers to the top-left corner of the image, w refers to the width and h refers the height. You should chat with me based on the fact in the image instead of imagination. Now I tell you the points with their visual description:\n{points_with_caps}\n. Now begin chatting! Human: {chat_input}\nAI: "
    # "The image is of width {width} and height {height}." 
    point_chat_prompt = "a) Revised prompt: I am an AI trained to chat with you about an image based on specific points (w, h) you provide, along with their visual descriptions. Please note that (0, 0) refers to the top-left corner of the image, w refers to the width, and h refers to the height. Here are the points and their descriptions you've given me: {points_with_caps}. Now, let's chat! Human: {chat_input} AI:"
    prev_visual_context = ""
    pos_points = [f"{points[i][0]}, {points[i][1]}" for i in range(len(points)) if labels[i] == 1]
    prev_visual_context = ', '.join(pos_points) + captions[-1] + '\n'
    chat_prompt = point_chat_prompt.format(**{"points_with_caps": prev_visual_context, "chat_input": chat_input})
    response = model.text_refiner.llm(chat_prompt)
    state = state + [(chat_input, response)]
    return state, state

def init_openai_api_key(api_key):
    # os.environ['OPENAI_API_KEY'] = api_key
    global model
    model = CaptionAnything(args, api_key)

css='''
#image_upload{min-height:200px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 200px}
'''

with gr.Blocks(css=css) as iface:
    state = gr.State([])
    click_state = gr.State([[],[],[]])
    caption_state = gr.State([[]])
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Column():
        openai_api_key = gr.Textbox(
            placeholder="Input your openAI API key and press Enter",
            show_label=False,
            lines=1,
            type="password",
        )
        openai_api_key.submit(init_openai_api_key, inputs=[openai_api_key])
        
        with gr.Row():
            with gr.Column(scale=0.7):
                image_input = gr.Image(type="pil", interactive=True, label="Image", elem_id="image_upload").style(height=260,scale=1.0)

                with gr.Row(scale=0.7):
                    point_prompt = gr.Radio(
                        choices=["Positive Point",  "Negative Point"],
                        value="Positive Point",
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
                    image_output_crop= gr.Image(type="pil", interactive=False, label="Cropped Image by Mask", show_progress=False).style(height=260,scale=1.0)
                chatbot = gr.Chatbot(label="Chat Output",).style(height=450,scale=0.5)
        
        with gr.Row():
            with gr.Column(scale=0.7):
                prompt_input = gr.Textbox(lines=1, label="Input Prompt (A list of points like : [[100, 200, 1], [200,300,0]])")
                prompt_input.submit(
                    inference_seg_cap,
                    [
                        image_input,
                        prompt_input,
                        language,
                        sentiment,
                        factuality,
                        length,
                        state,
                        click_state
                    ],
                    [chatbot, state, click_state, image_output_mask, image_output_crop],
                    show_progress=False
                )
                
                image_input.upload(
                    upload_callback,
                    [image_input, state],
                    [chatbot]
                    )

                with gr.Row():
                    clear_button = gr.Button(value="Clear Click", interactive=True)
                    clear_button.click(
                        lambda: ("", [[], [], []], None, None),
                        [],
                        [prompt_input, click_state, image_output_mask, image_output_crop],
                        queue=False,
                        show_progress=False
                    )
                    
                    clear_button = gr.Button(value="Clear", interactive=True)
                    clear_button.click(
                        lambda: ("", [], [], [[], [], []], None, None),
                        [],
                        [prompt_input, chatbot, state, click_state, image_output_mask, image_output_crop],
                        queue=False,
                        show_progress=False
                    )
                    
                    submit_button = gr.Button(
                        value="Submit", interactive=True, variant="primary"
                    )
                    submit_button.click(
                        inference_seg_cap,
                        [
                            image_input,
                            prompt_input,
                            language,
                            sentiment,
                            factuality,
                            length,
                            state,
                            click_state
                        ],
                        [chatbot, state, click_state, image_output_mask, image_output_crop],
                        show_progress=False
                    )
                    
                # select coordinate
                image_input.select(
                    get_select_coords, 
                    inputs=[image_input,point_prompt,language,sentiment,factuality,length,state,click_state], 
                    outputs=[prompt_input, chatbot, state, click_state, image_output_mask, image_output_crop],
                    show_progress=False
                    )

                image_input.change(
                    lambda: ("", [], [[], [], []]),
                    [],
                    [chatbot, state, click_state],
                    queue=False,
                )
                
            with gr.Column(scale=1.5):
                chat_input = gr.Textbox(lines=1, label="Chat Input")
                chat_input.submit(chat_with_points, [chat_input, click_state, state], [chatbot, state])
                
                    
    examples = gr.Examples(
        examples=examples,
        inputs=[image_input, prompt_input],
    )

iface.queue(concurrency_count=1, api_open=False, max_size=10)
iface.launch(server_name="0.0.0.0", enable_queue=True, server_port=args.port, share=args.gradio_share)
