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
import numpy as np
import PIL.ImageDraw as ImageDraw
from image_editing_utils import create_bubble_frame
import copy
from tools import mask_painter
from PIL import Image
import os
from captioner import build_captioner
from segment_anything import sam_model_registry
from text_refiner import build_text_refiner
from segmenter import build_segmenter

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

download_checkpoint(checkpoint_url, folder, filename)


title = """<h1 align="center">Caption-Anything</h1>"""
description = """Gradio demo for Caption Anything, image to dense captioning generation with various language styles. To use it, simply upload your image, or click one of the examples to load them. Code: https://github.com/ttengwang/Caption-Anything
"""

examples = [
    ["test_img/img35.webp"],
    ["test_img/img2.jpg"],
    ["test_img/img5.jpg"],
    ["test_img/img12.jpg"],
    ["test_img/img14.jpg"],
    ["test_img/img0.png"],
    ["test_img/img1.jpg"],
]

args = parse_augment()
# args.device = 'cuda:5'
# args.disable_gpt = True
# args.enable_reduce_tokens = False
# args.port=20322
# args.captioner = 'blip'
# args.regular_box = True
shared_captioner = build_captioner(args.captioner, args.device, args)
shared_sam_model = sam_model_registry['vit_h'](checkpoint=args.segmenter_checkpoint).to(args.device)


def build_caption_anything_with_models(args, api_key="", captioner=None, sam_model=None, text_refiner=None, session_id=None):
    segmenter = build_segmenter(args.segmenter, args.device, args, model=sam_model)
    captioner = captioner
    if session_id is not None:
        print('Init caption anything for session {}'.format(session_id))
    return CaptionAnything(args, api_key, captioner=captioner, segmenter=segmenter, text_refiner=text_refiner)


def init_openai_api_key(api_key=""):
    text_refiner = None
    if api_key and len(api_key) > 30:
        try:
            text_refiner = build_text_refiner(args.text_refiner, args.device, args, api_key)
            text_refiner.llm('hi') # test
        except:
            text_refiner = None
    openai_available = text_refiner is not None
    return gr.update(visible = openai_available), gr.update(visible = openai_available), gr.update(visible = openai_available), gr.update(visible = True), gr.update(visible = True), gr.update(visible = True), text_refiner


def get_prompt(chat_input, click_state, click_mode):
    inputs = json.loads(chat_input)
    if click_mode == 'Continuous':
        points = click_state[0]
        labels = click_state[1]
        for input in inputs:
            points.append(input[:2])
            labels.append(input[2])
    elif click_mode == 'Single':
        points = []
        labels = []
        for input in inputs:
            points.append(input[:2])
            labels.append(input[2])
        click_state[0] = points
        click_state[1] = labels
    else:
        raise NotImplementedError
    
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt

def update_click_state(click_state, caption, click_mode):
    if click_mode == 'Continuous':
        click_state[2].append(caption)
    elif click_mode == 'Single':
        click_state[2] = [caption]
    else:
        raise NotImplementedError     


def chat_with_points(chat_input, click_state, state, text_refiner):
    if text_refiner is None:
        response = "Text refiner is not initilzed, please input openai api key."
        state = state + [(chat_input, response)]
        return state, state
    
    points, labels, captions = click_state
    # point_chat_prompt = "I want you act as a chat bot in terms of image. I will give you some points (w, h) in the image and tell you what happed on the point in natural language. Note that (0, 0) refers to the top-left corner of the image, w refers to the width and h refers the height. You should chat with me based on the fact in the image instead of imagination. Now I tell you the points with their visual description:\n{points_with_caps}\nNow begin chatting! Human: {chat_input}\nAI: "
    # # "The image is of width {width} and height {height}." 
    point_chat_prompt = "a) Revised prompt: I am an AI trained to chat with you about an image based on specific points (w, h) you provide, along with their visual descriptions. Please note that (0, 0) refers to the top-left corner of the image, w refers to the width, and h refers to the height. Here are the points and their descriptions you've given me: {points_with_caps}. Now, let's chat! Human: {chat_input} AI:"
    prev_visual_context = ""
    pos_points = [f"{points[i][0]}, {points[i][1]}" for i in range(len(points)) if labels[i] == 1]
    if len(captions):
        prev_visual_context = ', '.join(pos_points) + captions[-1] + '\n'
    else:
        prev_visual_context = 'no point exists.'
    chat_prompt = point_chat_prompt.format(**{"points_with_caps": prev_visual_context, "chat_input": chat_input})
    response = text_refiner.llm(chat_prompt)
    state = state + [(chat_input, response)]
    return state, state

def inference_seg_cap(image_input, point_prompt, click_mode, language, sentiment, factuality, 
                length, image_embedding, state, click_state, original_size, input_size, text_refiner, evt:gr.SelectData):
    
    model = build_caption_anything_with_models(
        args,    
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        text_refiner=text_refiner,
        session_id=iface.app_id
    )
    
    model.segmenter.image_embedding = image_embedding
    model.segmenter.predictor.original_size = original_size
    model.segmenter.predictor.input_size = input_size
    model.segmenter.predictor.is_image_set = True

    if point_prompt == 'Positive':
        coordinate = "[[{}, {}, 1]]".format(str(evt.index[0]), str(evt.index[1]))
    else:
        coordinate = "[[{}, {}, 0]]".format(str(evt.index[0]), str(evt.index[1]))
        
    controls = {'length': length,
             'sentiment': sentiment,
             'factuality': factuality,
             'language': language}

    # click_coordinate = "[[{}, {}, 1]]".format(str(evt.index[0]), str(evt.index[1])) 
    # chat_input = click_coordinate
    prompt = get_prompt(coordinate, click_state, click_mode)
    print('prompt: ', prompt, 'controls: ', controls)

    out = model.inference(image_input, prompt, controls, disable_gpt=True)
    state = state + [(None, "Image point: {}, Input label: {}".format(prompt["input_point"], prompt["input_label"]))]
    # for k, v in out['generated_captions'].items():
    #     state = state + [(f'{k}: {v}', None)]
    state = state + [("caption: {}".format(out['generated_captions']['raw_caption']), None)]
    wiki = out['generated_captions'].get('wiki', "")

    update_click_state(click_state, out['generated_captions']['raw_caption'], click_mode)
    text = out['generated_captions']['raw_caption']
    # draw = ImageDraw.Draw(image_input)
    # draw.text((evt.index[0], evt.index[1]), text, textcolor=(0,0,255), text_size=120)
    input_mask = np.array(out['mask'].convert('P'))
    image_input = mask_painter(np.array(image_input), input_mask)
    origin_image_input = image_input
    image_input = create_bubble_frame(image_input, text, (evt.index[0], evt.index[1]))

    yield state, state, click_state, chat_input, image_input, wiki
    if not args.disable_gpt and model.text_refiner:
        refined_caption = model.text_refiner.inference(query=text, controls=controls, context=out['context_captions'])
        # new_cap = 'Original: ' + text + '. Refined: ' + refined_caption['caption']
        new_cap = refined_caption['caption']
        refined_image_input = create_bubble_frame(origin_image_input, new_cap, (evt.index[0], evt.index[1]))
        yield state, state, click_state, chat_input, refined_image_input, wiki


def upload_callback(image_input, state):
    state = [] + [('Image size: ' + str(image_input.size), None)]
    click_state = [[], [], []]
    res = 1024
    width, height = image_input.size
    ratio = min(1.0 * res / max(width, height), 1.0)
    if ratio < 1.0:
        image_input = image_input.resize((int(width * ratio), int(height * ratio)))
        print('Scaling input image to {}'.format(image_input.size))
        
    model = build_caption_anything_with_models(
        args,    
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        session_id=iface.app_id
    )
    model.segmenter.set_image(image_input)
    image_embedding = model.segmenter.image_embedding
    original_size = model.segmenter.predictor.original_size
    input_size = model.segmenter.predictor.input_size
    return state, state, image_input, click_state, image_input, image_embedding, original_size, input_size

with gr.Blocks(
    css='''
    #image_upload{min-height:400px}
    #image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 600px}
    '''
) as iface:
    state = gr.State([])
    click_state = gr.State([[],[],[]])
    origin_image = gr.State(None)
    image_embedding = gr.State(None)
    text_refiner = gr.State(None)
    original_size = gr.State(None)
    input_size = gr.State(None)

    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1.0):
            with gr.Column(visible=False) as modules_not_need_gpt:
                image_input = gr.Image(type="pil", interactive=True, elem_id="image_upload")
                example_image = gr.Image(type="pil", interactive=False, visible=False)
                with gr.Row(scale=1.0):
                    with gr.Row(scale=0.4):
                        point_prompt = gr.Radio(
                            choices=["Positive",  "Negative"],
                            value="Positive",
                            label="Point Prompt",
                            interactive=True)
                        click_mode = gr.Radio(
                            choices=["Continuous",  "Single"],
                            value="Continuous",
                            label="Clicking Mode",
                            interactive=True)
                    with gr.Row(scale=0.4):
                        clear_button_clike = gr.Button(value="Clear Clicks", interactive=True)
                        clear_button_image = gr.Button(value="Clear Image", interactive=True)
            with gr.Column(visible=False) as modules_need_gpt:
                with gr.Row(scale=1.0):
                    language = gr.Dropdown(['English', 'Chinese', 'French', "Spanish", "Arabic", "Portuguese", "Cantonese"], value="English", label="Language", interactive=True)

                    sentiment = gr.Radio(
                        choices=["Positive", "Natural", "Negative"],
                        value="Natural",
                        label="Sentiment",
                        interactive=True,
                    )
                with gr.Row(scale=1.0):
                    factuality = gr.Radio(
                        choices=["Factual", "Imagination"],
                        value="Factual",
                        label="Factuality",
                        interactive=True,
                    )
                    length = gr.Slider(
                        minimum=10,
                        maximum=80,
                        value=10,
                        step=1,
                        interactive=True,
                        label="Length",
                    )   
            with gr.Column(visible=True) as modules_not_need_gpt3:
                gr.Examples(
                    examples=examples,
                    inputs=[example_image],
                )
        with gr.Column(scale=0.5):
            openai_api_key = gr.Textbox(
                placeholder="Input openAI API key",
                show_label=False,
                label = "OpenAI API Key",
                lines=1,
                type="password")
            with gr.Row(scale=0.5):
                enable_chatGPT_button = gr.Button(value="Run with ChatGPT", interactive=True, variant='primary')
                disable_chatGPT_button = gr.Button(value="Run without ChatGPT (Faster)", interactive=True, variant='primary')
            with gr.Column(visible=False) as modules_need_gpt2:
                wiki_output = gr.Textbox(lines=5, label="Wiki", max_lines=5)
            with gr.Column(visible=False) as modules_not_need_gpt2:
                chatbot = gr.Chatbot(label="Chat about Selected Object",).style(height=550,scale=0.5)
                with gr.Column(visible=False) as modules_need_gpt3:
                    chat_input = gr.Textbox(lines=1, label="Chat Input")
                    with gr.Row():
                        clear_button_text = gr.Button(value="Clear Text", interactive=True)
                        submit_button_text = gr.Button(value="Submit", interactive=True, variant="primary")
    
    openai_api_key.submit(init_openai_api_key, inputs=[openai_api_key], outputs=[modules_need_gpt,modules_need_gpt2, modules_need_gpt3, modules_not_need_gpt, modules_not_need_gpt2, modules_not_need_gpt3, text_refiner])
    enable_chatGPT_button.click(init_openai_api_key, inputs=[openai_api_key], outputs=[modules_need_gpt,modules_need_gpt2, modules_need_gpt3, modules_not_need_gpt, modules_not_need_gpt2, modules_not_need_gpt3, text_refiner])
    disable_chatGPT_button.click(init_openai_api_key, outputs=[modules_need_gpt,modules_need_gpt2, modules_need_gpt3, modules_not_need_gpt, modules_not_need_gpt2, modules_not_need_gpt3, text_refiner])
    
    clear_button_clike.click(
        lambda x: ([[], [], []], x, ""),
        [origin_image],
        [click_state, image_input, wiki_output],
        queue=False,
        show_progress=False
    )
    clear_button_image.click(
        lambda: (None, [], [], [[], [], []], "", ""),
        [],
        [image_input, chatbot, state, click_state, wiki_output, origin_image],
        queue=False,
        show_progress=False
    )
    clear_button_text.click(
        lambda: ([], [], [[], [], []]),
        [],
        [chatbot, state, click_state],
        queue=False,
        show_progress=False
    )
    image_input.clear(
        lambda: (None, [], [], [[], [], []], "", ""),
        [],
        [image_input, chatbot, state, click_state, wiki_output, origin_image],
        queue=False,
        show_progress=False
    )

    image_input.upload(upload_callback,[image_input, state], [chatbot, state, origin_image, click_state, image_input, image_embedding, original_size, input_size])
    chat_input.submit(chat_with_points, [chat_input, click_state, state, text_refiner], [chatbot, state])
    example_image.change(upload_callback,[example_image, state], [state, state, origin_image, click_state, image_input, image_embedding, original_size, input_size])

    # select coordinate
    image_input.select(inference_seg_cap, 
        inputs=[
        origin_image,
        point_prompt,
        click_mode,
        language,
        sentiment,
        factuality,
        length,
        image_embedding,
        state,
        click_state,
        original_size, 
        input_size,
        text_refiner
        ],
        outputs=[chatbot, state, click_state, chat_input, image_input, wiki_output],
        show_progress=False, queue=True)
    
iface.queue(concurrency_count=1, api_open=False, max_size=10)
iface.launch(server_name="0.0.0.0", enable_queue=True, server_port=args.port, share=args.gradio_share)  