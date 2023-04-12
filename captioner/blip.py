import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import BlipProcessor
from .modeling_blip import BlipForConditionalGeneration
import json
import pdb
import cv2
import numpy as np
from typing import Union
from .base_captioner import BaseCaptioner
import torchvision.transforms.functional as F 


class BLIPCaptioner(BaseCaptioner):
    def __init__(self, device, enable_filter=False):
        super().__init__(device, enable_filter)
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=self.torch_dtype).to(self.device)
        
    @torch.no_grad()
    def inference(self, image: Union[np.ndarray, Image.Image, str], filter=False):
        if type(image) == str: # input path
                image = Image.open(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs, max_new_tokens=50)
        captions = self.processor.decode(out[0], skip_special_tokens=True).strip()
        if self.enable_filter and filter:
            captions = self.filter_caption(image, captions)
        print(f"\nProcessed ImageCaptioning by BLIPCaptioner, Output Text: {captions}")
        return captions
    
    @torch.no_grad()
    def inference_with_reduced_tokens(self, image: Union[np.ndarray, Image.Image, str], seg_mask, crop_mode="w_bg", filter=False, disable_regular_box = False):
        crop_save_path = self.generate_seg_cropped_image(image=image, seg_mask=seg_mask, crop_mode=crop_mode, disable_regular_box=disable_regular_box)
        if type(image) == str: # input path
            image = Image.open(image)
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device, self.torch_dtype)
        _, _, H, W = pixel_values.shape
        seg_mask = Image.fromarray(seg_mask.astype(float))
        seg_mask = seg_mask.resize((H, W))
        seg_mask = F.pil_to_tensor(seg_mask) > 0.5
        seg_mask = seg_mask.float()
        pixel_masks = seg_mask.unsqueeze(0).to(self.device)
        out = self.model.generate(pixel_values=pixel_values, pixel_masks=pixel_masks, max_new_tokens=50)
        captions = self.processor.decode(out[0], skip_special_tokens=True).strip()
        if self.enable_filter and filter:
            captions = self.filter_caption(image, captions)
        print(f"\nProcessed ImageCaptioning by BLIPCaptioner, Output Text: {captions}")
        return captions, crop_save_path


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
    print(model.inference_with_reduced_tokens(image_path, seg_mask))
    