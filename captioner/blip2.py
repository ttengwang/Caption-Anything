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
    def __init__(self, device, prompt: str = None, cache_dir = None):
        super().__init__(device)
        self.device = device
        self.prompt = prompt
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
        if self.prompt is not None:
            inputs = self.processor(image, text = self.prompt, return_tensors="pt").to(self.device, self.torch_dtype)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs, max_new_tokens=50)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning by BLIP2Captioner, Output Text: {captions}")
        return captions

if __name__ == '__main__':

    prompt = 'Question: what is the animal in the picture? Answer: a dog. Question: where is this dog and what is it doing? Answer: '
    model = BLIP2Captioner(device='cuda:4', prompt = prompt, cache_dir = '/nvme-ssd/fjj/Caption-Anything/model_cache')
    image_path = 'test_img/img2.jpg'
    seg_mask = np.zeros((224,224))
    seg_mask[50:200, 50:200] = 1
    print(f'process image {image_path}')
    print(model.inference_seg(image_path, seg_mask))