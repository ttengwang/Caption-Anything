import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import json
import pdb
import cv2
import numpy as np
from typing import Union


from base_captioner import BaseCaptioner

class BLIPCaptioner(BaseCaptioner):
    def __init__(self, device):
        super().__init__(device)
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=self.huggingface_cache_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype, cache_dir=self.huggingface_cache_dir).to(self.device)
        

    def inference(self, image: Union[np.ndarray, Image.Image, str]):
        if type(image) == str: # input path
                image = Image.open(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs, max_new_tokens=50)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning by BLIPCaptioner, Output Text: {captions}")
        return captions


if __name__ == '__main__':
    model = BLIPCaptioner(device='cuda:0')
    image_path = 'test_img/img2.jpg'
    seg_mask = np.zeros((15,15))
    seg_mask[5:10, 5:10] = 1
    print(f'process image {image_path}')
    print(model.inference_seg(image_path, seg_mask))
    