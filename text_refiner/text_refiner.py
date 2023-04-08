from langchain.llms.openai import OpenAI
import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

class TextRefiner:
    def __init__(self, device):
        print(f"Initializing TextRefiner to {device}")
        # self.device = device
        # self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.prompts = {'length': "convert the sentence within 10 words, do not change its meaning",
                        "sentiment": "convert the sentence to {sentiment}, do not change its meaning"
                        }
    def parse(self, response):
        out = response.strip()
        return out
    
    def prepare_input(self, prompts, query)
    
    def inference(self, query: str, controls: list):
        prompts = []
        for control in controls:
            prompts.append(self.prompts[control])
        input = self.prepare_input(query, prompts)
        response = self.llm(input)
        response = self.parse(response)            
        return response
    