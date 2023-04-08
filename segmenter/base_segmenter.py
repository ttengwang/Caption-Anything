import torch
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import Union

class BaseSegmenter:
    def __init__(self, device):
        print(f"Initializing BaseSegmenter to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = None
        self.model = None
        
    def inference(image: str, prompt: Any=None):
        if type(image) == str: # input path
            image = Image.open(image)
        
        # Implement segment anything
