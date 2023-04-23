# Copyright (c) Microsoft
# Modified from Visual ChatGPT Project https://github.com/microsoft/TaskMatrix/blob/main/visual_chatgpt.py

import os
import gradio as gr
import re
import uuid
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import argparse
import inspect

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

VISUAL_CHATGPT_PREFIX = """
    I want you act as Caption Anything Chatbox (short as CATchat), which is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. You are able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    As a language model, you can not directly read images, but can invoke VQA tool to indirectly understand pictures, by repeatly asking questions about the objects and scene of the image. You should carefully asking informative questions to maximize your information about this image content. Each image will have a file name formed as "chat_image/xxx.png", you are very strict to the file name and will never fabricate nonexistent files.
    
    You have access to the following tools:"""


# TOOLS:
# ------

# Visual ChatGPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

"Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}], remember the action must to be one tool
Action Input: the input to the action
Observation: the result of the action"

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

"Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]"

"""

VISUAL_CHATGPT_SUFFIX = """
Begin Chatting!

Previous conversation history:
{chat_history}

New input: {input}
As a language model, you must repeatly to use VQA tools to observe images. You response should be consistent with the outputs of the VQA tool instead of imagination. Do not repeat asking the same question. 

Thought: Do I need to use a tool? {agent_scratchpad} (You are strictly to use the aforementioned "Thought/Action/Action Input/Observation" format as the answer.)"""

os.makedirs('chat_image', exist_ok=True)


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator

def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)

def get_new_image_name(folder='chat_image', func_name="update"):
    this_new_uuid = str(uuid.uuid4())[:8]
    new_file_name = f'{func_name}_{this_new_uuid}.png'
    return os.path.join(folder, new_file_name)

class VisualQuestionAnswering:
    def __init__(self, device):
        print(f"Initializing VisualQuestionAnswering to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype).to(self.device)
        # self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        # self.model = BlipForQuestionAnswering.from_pretrained(
            # "Salesforce/blip-vqa-capfilt-large", torch_dtype=self.torch_dtype).to(self.device)

    @prompts(name="Answer Question About The Image",
             description="VQA tool is useful when you need an answer for a question based on an image. "
                         "like: what is the color of an object, how many cats in this figure, where is the child sitting, what does the cat doing, why is he laughing."
                         "The input to this tool should be a comma separated string of two, representing the image path and the question.")
    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
              f"Output Answer: {answer}")
        return answer
    
def build_chatbot_tools(load_dict):
    print(f"Initializing ChatBot, load_dict={load_dict}")
    models = {}
    # Load Basic Foundation Models
    for class_name, device in load_dict.items():
        models[class_name] = globals()[class_name](device=device)

    # Load Template Foundation Models
    for class_name, module in globals().items():
        if getattr(module, 'template_model', False):
            template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
            loaded_names = set([type(e).__name__ for e in models.values()])
            if template_required_names.issubset(loaded_names):
                models[class_name] = globals()[class_name](
                    **{name: models[name] for name in template_required_names})
                
    tools = []
    for instance in models.values():
        for e in dir(instance):
            if e.startswith('inference'):
                func = getattr(instance, e)
                tools.append(Tool(name=func.name, description=func.description, func=func))
    return tools

class ConversationBot:
    def __init__(self, tools, api_key=""):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=api_key)
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = tools
        self.current_image = None
        self.point_prompt = ""
        self.global_prompt = ""
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': VISUAL_CHATGPT_PREFIX, 'format_instructions': VISUAL_CHATGPT_FORMAT_INSTRUCTIONS,
                          'suffix': VISUAL_CHATGPT_SUFFIX}, )

    def constructe_intermediate_steps(self, agent_res):
        ans = []
        for action, output in agent_res:
            if hasattr(action, "tool_input"):
                use_tool = "Yes"
                act = (f"Thought: Do I need to use a tool? {use_tool}\nAction: {action.tool}\nAction Input: {action.tool_input}", f"Observation: {output}")
            else:
                use_tool = "No"
                act = (f"Thought: Do I need to use a tool? {use_tool}", f"AI: {output}")
            act= list(map(lambda x: x.replace('\n', '<br>'), act))
            ans.append(act)
        return ans
    
    def run_text(self, text, state, aux_state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        if self.point_prompt != "":
            Human_prompt = f'\nHuman: {self.point_prompt}\n'
            AI_prompt = 'Ok'
            self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
            self.point_prompt = ""
        res = self.agent({"input": text})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(chat_image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        
        aux_state = aux_state + [(f"User Input: {text}", None)]
        aux_state = aux_state + self.constructe_intermediate_steps(res['intermediate_steps'])
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}\n"
              f"Aux state: {aux_state}\n"
              )
        return state, state, aux_state, aux_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default="VisualQuestionAnswering_cuda:0")
    parser.add_argument('--port', type=int, default=1015)
    
    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    tools = build_chatbot_tools(load_dict)
    bot = ConversationBot(tools)
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        with gr.Row():
            chatbot = gr.Chatbot(elem_id="chatbot", label="CATchat").style(height=1000,scale=0.5)
            auxwindow = gr.Chatbot(elem_id="chatbot", label="Aux Window").style(height=1000,scale=0.5)
        state = gr.State([])
        aux_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["image"])

        txt.submit(bot.run_text, [txt, state, aux_state], [chatbot, state, aux_state, auxwindow])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt, aux_state], [chatbot, state, txt, aux_state, auxwindow])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, auxwindow)
        clear.click(lambda: [], None, state)
        clear.click(lambda: [], None, aux_state)
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)
