import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import json
import pdb
import cv2
import numpy as np
from typing import Union
from captioner.base_captioner import BaseCaptioner

class BLIPCaptioner(BaseCaptioner):
    def __init__(self, device):
        super().__init__(device)
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype).to(self.device)

if __name__ == '__main__':
    model = BLIPCaptioner(device='cuda:0')
    # image_path = 'test_img/img2.jpg'
    image_path = '/group/30042/wybertwang/project/woa_visgpt/chatARC/image/SAM/img10.jpg'
    seg_mask = np.zeros((15,15))
    seg_mask[5:10, 5:10] = 1
    seg_mask = 'test_img/img10.jpg.raw_mask.png'
    image_path = 'test_img/img2.jpg'
    seg_mask = 'test_img/img2.jpg.raw_mask.png'
    print(f'process image {image_path}')
    print(model.inference_seg(image_path, seg_mask))
    