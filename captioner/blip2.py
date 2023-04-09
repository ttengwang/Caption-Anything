import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import json
import pdb
import cv2
import numpy as np
from typing import Union
from base_captioner import BaseCaptioner

class BLIP2Captioner(BaseCaptioner):
    def __init__(self, device, dialogue: bool = False, cache_dir = None):
        super().__init__(device)
        self.device = device
        self.dialogue = dialogue
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        if cache_dir is not None:
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir = cache_dir)
            self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir = cache_dir, torch_dtype = self.torch_dtype).to(device)
        else:
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype = self.torch_dtype).to(device)
        
    @torch.no_grad()
    def inference(self, image: Union[np.ndarray, Image.Image, str]):
        if type(image) == str: # input path
                image = Image.open(image)

        if not self.dialogue:
            inputs = self.processor(image, return_tensors="pt").to(self.device, self.torch_dtype)
            out = self.model.generate(**inputs, max_new_tokens=50)
            captions = self.processor.decode(out[0], skip_special_tokens=True)
            similarity = self.filter_caption(image, captions)
            if similarity < self.threshold:
                print('There seems to be nothing where you clicked.')
                return ''
            print(f"\nProcessed ImageCaptioning by BLIP2Captioner, Output Text: {captions}")
            return captions
        else:
            context = []
            template = "Question: {} Answer: {}."
            while(True):
                input_texts = input()
                if input_texts == 'end':
                    break
                prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + input_texts + " Answer:"
                inputs = self.processor(image, text = prompt, return_tensors="pt").to(self.device, self.torch_dtype)
                out = self.model.generate(**inputs, max_new_tokens=50)
                captions = self.processor.decode(out[0], skip_special_tokens=True).strip()
                context.append((input_texts, captions))
    
        return captions

if __name__ == '__main__':

    dialogue = False
    model = BLIP2Captioner(device='cuda:4', dialogue = dialogue, cache_dir = '/nvme-ssd/fjj/Caption-Anything/model_cache')
    image_path = 'test_img/img2.jpg'
    seg_mask = np.zeros((224,224))
    seg_mask[50:200, 50:200] = 1
    print(f'process image {image_path}')
    print(model.inference_seg(image_path, seg_mask))