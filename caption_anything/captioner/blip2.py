import torch
from PIL import Image
import numpy as np
from typing import Union
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from caption_anything.utils.utils import is_platform_win, load_image
from .base_captioner import BaseCaptioner

class BLIP2Captioner(BaseCaptioner):
    def __init__(self, device, dialogue: bool = False, enable_filter: bool = False):
        super().__init__(device, enable_filter)
        self.device = device
        self.dialogue = dialogue
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if is_platform_win():
            self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="sequential", torch_dtype=self.torch_dtype)
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map='sequential', load_in_8bit=True)

    @torch.no_grad()
    def inference(self, image: Union[np.ndarray, Image.Image, str], filter=False, **kargs):
        image = load_image(image, return_type="pil")

        if not self.dialogue:
            text_prompt = kargs.get('text_prompt', 'Question: what does the image show? Answer:')
            inputs = self.processor(image, text = text_prompt, return_tensors="pt").to(self.device, self.torch_dtype)
            out = self.model.generate(**inputs, max_new_tokens=50)
            captions = self.processor.decode(out[0], skip_special_tokens=True).strip()
            if self.enable_filter and filter:
                captions = self.filter_caption(image, captions)
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
    image_path = 'test_images/img2.jpg'
    seg_mask = np.zeros((224,224))
    seg_mask[50:200, 50:200] = 1
    print(f'process image {image_path}')
    print(model.inference_seg(image_path, seg_mask))
