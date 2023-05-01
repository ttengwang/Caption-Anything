import os
import json
import gradio as gr
import numpy as np
from gradio import processing_utils

from packaging import version
from PIL import Image, ImageDraw
import functools

from caption_anything.model import CaptionAnything
from caption_anything.utils.image_editing_utils import create_bubble_frame
from caption_anything.utils.utils import mask_painter, seg_model_map, prepare_segmenter, image_resize
from caption_anything.utils.parser import parse_augment
from caption_anything.captioner import build_captioner
from caption_anything.text_refiner import build_text_refiner
from caption_anything.segmenter import build_segmenter
from caption_anything.utils.chatbot import ConversationBot, build_chatbot_tools, get_new_image_name
from segment_anything import sam_model_registry
import easyocr

args = parse_augment()
if args.segmenter_checkpoint is None:
    _, segmenter_checkpoint = prepare_segmenter(args.segmenter)
else:
    segmenter_checkpoint = args.segmenter_checkpoint
    
shared_captioner = build_captioner(args.captioner, args.device, args)
shared_sam_model = sam_model_registry[seg_model_map[args.segmenter]](checkpoint=segmenter_checkpoint).to(args.device)
ocr_lang = ["ch_tra", "en"]
shared_ocr_reader = easyocr.Reader(ocr_lang)
tools_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.chat_tools_dict.split(',')}
shared_chatbot_tools = build_chatbot_tools(tools_dict)


class ImageSketcher(gr.Image):
    """
    Fix the bug of gradio.Image that cannot upload with tool == 'sketch'.
    """

    is_template = True  # Magic to make this work with gradio.Block, don't remove unless you know what you're doing.

    def __init__(self, **kwargs):
        super().__init__(tool="sketch", **kwargs)

    def preprocess(self, x):
        if self.tool == 'sketch' and self.source in ["upload", "webcam"]:
            assert isinstance(x, dict)
            if x['mask'] is None:
                decode_image = processing_utils.decode_base64_to_image(x['image'])
                width, height = decode_image.size
                mask = np.zeros((height, width, 4), dtype=np.uint8)
                mask[..., -1] = 255
                mask = self.postprocess(mask)
                x['mask'] = mask
        return super().preprocess(x)


def build_caption_anything_with_models(args, api_key="", captioner=None, sam_model=None, ocr_reader=None, text_refiner=None,
                                       session_id=None):
    segmenter = build_segmenter(args.segmenter, args.device, args, model=sam_model)
    captioner = captioner
    if session_id is not None:
        print('Init caption anything for session {}'.format(session_id))
    return CaptionAnything(args, api_key, captioner=captioner, segmenter=segmenter, ocr_reader=ocr_reader, text_refiner=text_refiner)


def init_openai_api_key(api_key=""):
    text_refiner = None
    visual_chatgpt = None
    if api_key and len(api_key) > 30:
        try:
            text_refiner = build_text_refiner(args.text_refiner, args.device, args, api_key)
            assert len(text_refiner.llm('hi')) > 0 # test
            visual_chatgpt = ConversationBot(shared_chatbot_tools, api_key)
        except:
            text_refiner = None
            visual_chatgpt = None
    openai_available = text_refiner is not None
    if openai_available:
        return [gr.update(visible=True)]*6 + [gr.update(visible=False)]*2 + [text_refiner, visual_chatgpt, None]
    else:
        return [gr.update(visible=False)]*6 + [gr.update(visible=True)]*2 + [text_refiner, visual_chatgpt, 'Your OpenAI API Key is not available']
        
def init_wo_openai_api_key():
        return  [gr.update(visible=False)]*4 + [gr.update(visible=True)]*2 + [gr.update(visible=False)]*2 + [None, None, None]

def get_click_prompt(chat_input, click_state, click_mode):
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
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }
    return prompt


def update_click_state(click_state, caption, click_mode):
    if click_mode == 'Continuous':
        click_state[2].append(caption)
    elif click_mode == 'Single':
        click_state[2] = [caption]
    else:
        raise NotImplementedError

def chat_input_callback(*args):
    visual_chatgpt, chat_input, click_state, state, aux_state = args
    if visual_chatgpt is not None:
        return visual_chatgpt.run_text(chat_input, state, aux_state)
    else:
        response = "Text refiner is not initilzed, please input openai api key."
        state = state + [(chat_input, response)]
        return state, state


        
def upload_callback(image_input, state, visual_chatgpt=None):
    
    if isinstance(image_input, dict):  # if upload from sketcher_input, input contains image and mask
        image_input, mask = image_input['image'], image_input['mask']

    click_state = [[], [], []]
    image_input = image_resize(image_input, res=1024)
        
    model = build_caption_anything_with_models(
        args,
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        ocr_reader=shared_ocr_reader,
        session_id=iface.app_id
    )
    model.segmenter.set_image(image_input)
    image_embedding = model.image_embedding
    original_size = model.original_size
    input_size = model.input_size
    
    if visual_chatgpt is not None:
        print('upload_callback: add caption to chatGPT memory')
        new_image_path = get_new_image_name('chat_image', func_name='upload')
        image_input.save(new_image_path)
        visual_chatgpt.current_image = new_image_path
        img_caption = model.captioner.inference(image_input, filter=False, args={'text_prompt':''})['caption']
        Human_prompt = f'\nHuman: The description of the image with path {new_image_path} is: {img_caption}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = "Received."
        visual_chatgpt.global_prompt = Human_prompt + 'AI: ' + AI_prompt
        visual_chatgpt.agent.memory.buffer = visual_chatgpt.agent.memory.buffer + visual_chatgpt.global_prompt 
    state = [(None, 'Received new image, resize it to width {} and height {}: '.format(image_input.size[0], image_input.size[1]))]

    return state, state, image_input, click_state, image_input, image_input, image_embedding, \
        original_size, input_size


def inference_click(image_input, point_prompt, click_mode, enable_wiki, language, sentiment, factuality,
                    length, image_embedding, state, click_state, original_size, input_size, text_refiner, visual_chatgpt,
                    evt: gr.SelectData):
    click_index = evt.index

    if point_prompt == 'Positive':
        coordinate = "[[{}, {}, 1]]".format(str(click_index[0]), str(click_index[1]))
    else:
        coordinate = "[[{}, {}, 0]]".format(str(click_index[0]), str(click_index[1]))

    prompt = get_click_prompt(coordinate, click_state, click_mode)
    input_points = prompt['input_point']
    input_labels = prompt['input_label']

    controls = {'length': length,
                'sentiment': sentiment,
                'factuality': factuality,
                'language': language}

    model = build_caption_anything_with_models(
        args,
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        ocr_reader=shared_ocr_reader,
        text_refiner=text_refiner,
        session_id=iface.app_id
    )

    model.setup(image_embedding, original_size, input_size, is_image_set=True)

    enable_wiki = True if enable_wiki in ['True', 'TRUE', 'true', True, 'Yes', 'YES', 'yes'] else False
    out = model.inference(image_input, prompt, controls, disable_gpt=True, enable_wiki=enable_wiki, verbose=True, args={'clip_filter': False})[0]

    state = state + [("Image point: {}, Input label: {}".format(prompt["input_point"], prompt["input_label"]), None)]
    state = state + [(None, "raw_caption: {}".format(out['generated_captions']['raw_caption']))]
    update_click_state(click_state, out['generated_captions']['raw_caption'], click_mode)
    text = out['generated_captions']['raw_caption']
    input_mask = np.array(out['mask'].convert('P'))
    image_input = mask_painter(np.array(image_input), input_mask)
    origin_image_input = image_input
    image_input = create_bubble_frame(image_input, text, (click_index[0], click_index[1]), input_mask,
                                      input_points=input_points, input_labels=input_labels)
    x, y = input_points[-1]
    
    if visual_chatgpt is not None:
        print('inference_click: add caption to chatGPT memory')
        new_crop_save_path = get_new_image_name('chat_image', func_name='crop')
        Image.open(out["crop_save_path"]).save(new_crop_save_path)
        point_prompt = f'You should primarly use tools on the selected regional image (description: {text}, path: {new_crop_save_path}), which is a part of the whole image (path: {visual_chatgpt.current_image}). If human mentioned some objects not in the selected region, you can use tools on the whole image.'
        visual_chatgpt.point_prompt = point_prompt

    yield state, state, click_state, image_input
    if not args.disable_gpt and model.text_refiner:
        refined_caption = model.text_refiner.inference(query=text, controls=controls, context=out['context_captions'],
                                                       enable_wiki=enable_wiki)
        # new_cap = 'Original: ' + text + '. Refined: ' + refined_caption['caption']
        new_cap = refined_caption['caption']
        if refined_caption['wiki']:
            state = state + [(None, "Wiki: {}".format(refined_caption['wiki']))]
        state = state + [(None, f"caption: {new_cap}")]
        refined_image_input = create_bubble_frame(origin_image_input, new_cap, (click_index[0], click_index[1]),
                                                  input_mask,
                                                  input_points=input_points, input_labels=input_labels)
        yield state, state, click_state, refined_image_input


def get_sketch_prompt(mask: Image.Image):
    """
    Get the prompt for the sketcher.
    TODO: This is a temporary solution. We should cluster the sketch and get the bounding box of each cluster.
    """

    mask = np.asarray(mask)[..., 0]

    # Get the bounding box of the sketch
    y, x = np.where(mask != 0)
    x1, y1 = np.min(x), np.min(y)
    x2, y2 = np.max(x), np.max(y)

    prompt = {
        'prompt_type': ['box'],
        'input_boxes': [
            [x1, y1, x2, y2]
        ]
    }

    return prompt


def inference_traject(sketcher_image, enable_wiki, language, sentiment, factuality, length, image_embedding, state,
                      original_size, input_size, text_refiner):
    image_input, mask = sketcher_image['image'], sketcher_image['mask']

    prompt = get_sketch_prompt(mask)
    boxes = prompt['input_boxes']

    controls = {'length': length,
                'sentiment': sentiment,
                'factuality': factuality,
                'language': language}

    model = build_caption_anything_with_models(
        args,
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        ocr_reader=shared_ocr_reader,
        text_refiner=text_refiner,
        session_id=iface.app_id
    )

    model.setup(image_embedding, original_size, input_size, is_image_set=True)

    enable_wiki = True if enable_wiki in ['True', 'TRUE', 'true', True, 'Yes', 'YES', 'yes'] else False
    out = model.inference(image_input, prompt, controls, disable_gpt=True, enable_wiki=enable_wiki)[0]

    # Update components and states
    state.append((f'Box: {boxes}', None))
    state.append((None, f'raw_caption: {out["generated_captions"]["raw_caption"]}'))
    text = out['generated_captions']['raw_caption']
    input_mask = np.array(out['mask'].convert('P'))
    image_input = mask_painter(np.array(image_input), input_mask)

    origin_image_input = image_input

    fake_click_index = (int((boxes[0][0] + boxes[0][2]) / 2), int((boxes[0][1] + boxes[0][3]) / 2))
    image_input = create_bubble_frame(image_input, text, fake_click_index, input_mask)

    yield state, state, image_input

    if not args.disable_gpt and model.text_refiner:
        refined_caption = model.text_refiner.inference(query=text, controls=controls, context=out['context_captions'],
                                                       enable_wiki=enable_wiki)

        new_cap = refined_caption['caption']
        if refined_caption['wiki']:
            state = state + [(None, "Wiki: {}".format(refined_caption['wiki']))]
        state = state + [(None, f"caption: {new_cap}")]
        refined_image_input = create_bubble_frame(origin_image_input, new_cap, fake_click_index, input_mask)

        yield state, state, refined_image_input

def clear_chat_memory(visual_chatgpt, keep_global=False):
    if visual_chatgpt is not None:
        visual_chatgpt.memory.clear()
        visual_chatgpt.point_prompt = ""
        if keep_global:
            visual_chatgpt.agent.memory.buffer = visual_chatgpt.global_prompt
        else:
            visual_chatgpt.current_image = None
            visual_chatgpt.global_prompt = ""

def cap_everything(image_input, visual_chatgpt, text_refiner):
    
    model = build_caption_anything_with_models(
        args,
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        ocr_reader=shared_ocr_reader,
        text_refiner=text_refiner,
        session_id=iface.app_id
    )
    paragraph = model.inference_cap_everything(image_input, verbose=True)
    # state = state + [(None, f"Caption Everything: {paragraph}")]  
    Human_prompt = f'\nThe description of the image with path {visual_chatgpt.current_image} is:\n{paragraph}\nThis information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
    AI_prompt = "Received."
    visual_chatgpt.global_prompt = Human_prompt + 'AI: ' + AI_prompt
    visual_chatgpt.agent.memory.buffer = visual_chatgpt.agent.memory.buffer + visual_chatgpt.global_prompt 
    return paragraph

    
def get_style():
    current_version = version.parse(gr.__version__)
    if current_version <= version.parse('3.24.1'):
        style = '''
        #image_sketcher{min-height:500px}
        #image_sketcher [data-testid="image"], #image_sketcher [data-testid="image"] > div{min-height: 500px}
        #image_upload{min-height:500px}
        #image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 500px}
        '''
    elif current_version <= version.parse('3.27'):
        style = '''
        #image_sketcher{min-height:500px}
        #image_upload{min-height:500px}
        '''
    else:
        style = None

    return style


def create_ui():
    title = """<p><h1 align="center">Caption-Anything</h1></p>
    """
    description = """<p>Gradio demo for Caption Anything, image to dense captioning generation with various language styles. To use it, simply upload your image, or click one of the examples to load them. Code: <a href="https://github.com/ttengwang/Caption-Anything">https://github.com/ttengwang/Caption-Anything</a> <a href="https://huggingface.co/spaces/TencentARC/Caption-Anything?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>"""

    examples = [
        ["test_images/img35.webp"],
        ["test_images/img2.jpg"],
        ["test_images/img5.jpg"],
        ["test_images/img12.jpg"],
        ["test_images/img14.jpg"],
        ["test_images/qingming3.jpeg"],
        ["test_images/img1.jpg"],
    ]

    with gr.Blocks(
            css=get_style()
    ) as iface:
        state = gr.State([])
        click_state = gr.State([[], [], []])
        # chat_state = gr.State([])
        origin_image = gr.State(None)
        image_embedding = gr.State(None)
        text_refiner = gr.State(None)
        visual_chatgpt = gr.State(None)
        original_size = gr.State(None)
        input_size = gr.State(None)
        # img_caption = gr.State(None)
        aux_state = gr.State([])

        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1.0):
                with gr.Column(visible=False) as modules_not_need_gpt:
                    with gr.Tab("Click"):
                        image_input = gr.Image(type="pil", interactive=True, elem_id="image_upload")
                        example_image = gr.Image(type="pil", interactive=False, visible=False)
                        with gr.Row(scale=1.0):
                            with gr.Row(scale=0.4):
                                point_prompt = gr.Radio(
                                    choices=["Positive", "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    interactive=True)
                                click_mode = gr.Radio(
                                    choices=["Continuous", "Single"],
                                    value="Continuous",
                                    label="Clicking Mode",
                                    interactive=True)
                            with gr.Row(scale=0.4):
                                clear_button_click = gr.Button(value="Clear Clicks", interactive=True)
                                clear_button_image = gr.Button(value="Clear Image", interactive=True)
                    with gr.Tab("Trajectory (beta)"):
                        sketcher_input = ImageSketcher(type="pil", interactive=True, brush_radius=20,
                                                       elem_id="image_sketcher")
                        with gr.Row():
                            submit_button_sketcher = gr.Button(value="Submit", interactive=True)

                with gr.Column(visible=False) as modules_need_gpt1:
                    with gr.Row(scale=1.0):
                        language = gr.Dropdown(
                            ['English', 'Chinese', 'French', "Spanish", "Arabic", "Portuguese", "Cantonese"],
                            value="English", label="Language", interactive=True)
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
                            label="Generated Caption Length",
                        )
                        enable_wiki = gr.Radio(
                            choices=["Yes", "No"],
                            value="No",
                            label="Enable Wiki",
                            interactive=True)
                # with gr.Column(visible=True) as modules_not_need_gpt3:
                gr.Examples(
                    examples=examples,
                    inputs=[example_image],
                )
            with gr.Column(scale=0.5):
                with gr.Column(visible=True) as module_key_input:
                    openai_api_key = gr.Textbox(
                        placeholder="Input openAI API key",
                        show_label=False,
                        label="OpenAI API Key",
                        lines=1,
                        type="password")
                    with gr.Row(scale=0.5):
                        enable_chatGPT_button = gr.Button(value="Run with ChatGPT", interactive=True, variant='primary')
                        disable_chatGPT_button = gr.Button(value="Run without ChatGPT (Faster)", interactive=True,
                                                        variant='primary')
                with gr.Column(visible=False) as module_notification_box:
                    notification_box = gr.Textbox(lines=1, label="Notification", max_lines=5, show_label=False)
                with gr.Column(visible=False) as modules_need_gpt2: 
                    paragraph_output = gr.Textbox(lines=7, label="Describe Everything", max_lines=7)
                with gr.Column(visible=False) as modules_need_gpt0:
                    cap_everything_button = gr.Button(value="Caption Everything in a Paragraph", interactive=True)
                with gr.Column(visible=False) as modules_not_need_gpt2: 
                    chatbot = gr.Chatbot(label="Chatbox", ).style(height=550, scale=0.5)
                    with gr.Column(visible=False) as modules_need_gpt3:
                        chat_input = gr.Textbox(show_label=False, placeholder="Enter text and press Enter").style(
                            container=False)
                        with gr.Row():
                            clear_button_text = gr.Button(value="Clear Text", interactive=True)
                            submit_button_text = gr.Button(value="Submit", interactive=True, variant="primary")

        openai_api_key.submit(init_openai_api_key, inputs=[openai_api_key],
                              outputs=[modules_need_gpt0, modules_need_gpt1, modules_need_gpt2, modules_need_gpt3, modules_not_need_gpt,
                                       modules_not_need_gpt2, module_key_input, module_notification_box, text_refiner, visual_chatgpt, notification_box])
        enable_chatGPT_button.click(init_openai_api_key, inputs=[openai_api_key],
                                    outputs=[modules_need_gpt0, modules_need_gpt1, modules_need_gpt2, modules_need_gpt3,
                                             modules_not_need_gpt,
                                             modules_not_need_gpt2, module_key_input, module_notification_box, text_refiner, visual_chatgpt, notification_box])
        disable_chatGPT_button.click(init_wo_openai_api_key,
                                     outputs=[modules_need_gpt0, modules_need_gpt1, modules_need_gpt2, modules_need_gpt3,
                                              modules_not_need_gpt,
                                              modules_not_need_gpt2, module_key_input, module_notification_box, text_refiner, visual_chatgpt, notification_box])
        
        enable_chatGPT_button.click(
            lambda: (None, [], [], [[], [], []], "", "", ""),
            [],
            [image_input, chatbot, state, click_state, paragraph_output, origin_image],
            queue=False,
            show_progress=False
        )
        openai_api_key.submit(
            lambda: (None, [], [], [[], [], []], "", "", ""),
            [],
            [image_input, chatbot, state, click_state, paragraph_output, origin_image],
            queue=False,
            show_progress=False
        )

        cap_everything_button.click(cap_everything, [origin_image, visual_chatgpt, text_refiner], [paragraph_output])
        
        clear_button_click.click(
            lambda x: ([[], [], []], x),
            [origin_image],
            [click_state, image_input],
            queue=False,
            show_progress=False
        )
        clear_button_click.click(functools.partial(clear_chat_memory, keep_global=True), inputs=[visual_chatgpt])
        clear_button_image.click(
            lambda: (None, [], [], [[], [], []], "", "", ""),
            [],
            [image_input, chatbot, state, click_state, paragraph_output, origin_image],
            queue=False,
            show_progress=False
        )
        clear_button_image.click(clear_chat_memory, inputs=[visual_chatgpt])
        clear_button_text.click(
            lambda: ([], [], [[], [], [], []]),
            [],
            [chatbot, state, click_state],
            queue=False,
            show_progress=False
        )
        clear_button_text.click(clear_chat_memory, inputs=[visual_chatgpt])
        
        image_input.clear(
            lambda: (None, [], [], [[], [], []], "", "", ""),
            [],
            [image_input, chatbot, state, click_state, paragraph_output, origin_image],
            queue=False,
            show_progress=False
        )

        image_input.clear(clear_chat_memory, inputs=[visual_chatgpt])
        

        image_input.upload(upload_callback, [image_input, state, visual_chatgpt],
                           [chatbot, state, origin_image, click_state, image_input, sketcher_input,
                            image_embedding, original_size, input_size])
        sketcher_input.upload(upload_callback, [sketcher_input, state, visual_chatgpt],
                              [chatbot, state, origin_image, click_state, image_input, sketcher_input,
                               image_embedding, original_size, input_size])
        chat_input.submit(chat_input_callback, [visual_chatgpt, chat_input, click_state, state, aux_state],
                          [chatbot, state, aux_state])
        chat_input.submit(lambda: "", None, chat_input)
        submit_button_text.click(chat_input_callback, [visual_chatgpt, chat_input, click_state, state, aux_state],
                          [chatbot, state, aux_state])
        submit_button_text.click(lambda: "", None, chat_input)
        example_image.change(upload_callback, [example_image, state, visual_chatgpt],
                             [chatbot, state, origin_image, click_state, image_input, sketcher_input,
                              image_embedding, original_size, input_size])
        example_image.change(clear_chat_memory, inputs=[visual_chatgpt])
        # select coordinate
        image_input.select(
            inference_click,
            inputs=[
                origin_image, point_prompt, click_mode, enable_wiki, language, sentiment, factuality, length,
                image_embedding, state, click_state, original_size, input_size, text_refiner, visual_chatgpt
            ],
            outputs=[chatbot, state, click_state, image_input],
            show_progress=False, queue=True
        )

        submit_button_sketcher.click(
            inference_traject,
            inputs=[
                sketcher_input, enable_wiki, language, sentiment, factuality, length, image_embedding, state,
                original_size, input_size, text_refiner
            ],
            outputs=[chatbot, state, sketcher_input],
            show_progress=False, queue=True
        )

        return iface


if __name__ == '__main__':
    iface = create_ui()
    iface.queue(concurrency_count=5, api_open=False, max_size=10)
    iface.launch(server_name="0.0.0.0", enable_queue=True, server_port=args.port, share=args.gradio_share)
