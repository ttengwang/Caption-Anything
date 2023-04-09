from transformers import GitForCausalLM, GitProcessor, AutoProcessor
from PIL import Image
import torch
from base_captioner import BaseCaptioner
import numpy as np
from typing import Union


class GITCaptioner(BaseCaptioner):
    def __init__(self, device):
        super().__init__(device)
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = AutoProcessor.from_pretrained("microsoft/git-large")
        self.model = GitForCausalLM.from_pretrained(
            "microsoft/git-large", torch_dtype=self.torch_dtype).to(self.device)
    
    @torch.no_grad()
    def inference(self, image: Union[np.ndarray, Image.Image, str]):
        if type(image) == str: # input path
            image = Image.open(image)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(pixel_values=pixel_values, max_new_tokens=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"\nProcessed ImageCaptioning by GITCaptioner, Output Text: {generated_caption}")
        return generated_caption


if __name__ == '__main__':
    model = GITCaptioner(device='cuda:0')
    image_path = 'test_img/img2.jpg'
    seg_mask = np.zeros((224,224))
    seg_mask[50:200, 50:200] = 1
    print(f'process image {image_path}')
    print(model.inference_seg(image_path, seg_mask))