import time
import torch
import cv2
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import Union
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import PIL

class BaseSegmenter:
    def __init__(self, device, checkpoint, model_type='vit_h'):
        print(f"Initializing BaseSegmenter to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = None
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.model.to(device=self.device)

        self.predictor = SamPredictor(self.model)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        self.image_embedding = None
        self.image = None
    
    @torch.no_grad()
    def inference(self, image, control):
        # Implement segment anything
        reuse_feature = False
        if self.image == image:
            reuse_feature = True
        else:
            self.image = image
        
        if type(image) == str: # input path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif type(image) == PIL.Image.Image:
            image = np.array(image)
        
        if 'everything' in control['prompt_type']:
            masks = self.mask_generator.generate(image)
            new_masks = np.concatenate([mask["segmentation"][np.newaxis,:] for mask in masks])
            return new_masks
        else:
            if not reuse_feature:
                self.predictor.set_image(image)
                self.image_embedding = self.predictor.get_image_embedding()
            else:
                self.predictor.features = self.image_embedding
      
        if 'mutimask_output' in control:
            masks, scores, logits = self.predictor.predict(
                point_coords = np.array(control['input_point']),
                point_labels = np.array(control['input_label']),
                multimask_output = True,
            )
        elif 'input_boxes' in control:
            transformed_boxes = self.predictor.transform.apply_boxes_torch(
                torch.tensor(control["input_boxes"], device=self.predictor.device),
                image.shape[:2]
            )
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.squeeze(1).cpu().numpy()
            
        else:
            input_point = np.array(control['input_point']) if 'click' in control['prompt_type'] else None
            input_label = np.array(control['input_label']) if 'click' in control['prompt_type'] else None
            input_box = np.array(control['input_box']) if 'box' in control['prompt_type'] else None
           
            masks, scores, logits = self.predictor.predict(
                point_coords = input_point,
                point_labels = input_label,
                box = input_box,
                multimask_output = False,
            )
            
            if 0 in control['input_label']:
                mask_input = logits[np.argmax(scores), :, :]
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box = input_box,
                    mask_input=mask_input[None, :, :],
                    multimask_output=False,
                )
  
        return masks

if __name__ == "__main__":
    image_path = 'segmenter/images/truck.jpg'
    prompts = [
        {
            "prompt_type":["click"],
            "input_point":[[500, 375]],
            "input_label":[1],
            "multimask_output":"True",
        },
        {
            "prompt_type":["click"],
            "input_point":[[500, 375], [1125, 625]],
            "input_label":[1, 0],
        },
        {
            "prompt_type":["click", "box"],
            "input_box":[425, 600, 700, 875],
            "input_point":[[575, 750]],
            "input_label": [0]
        },
        {
            "prompt_type":["box"],
            "input_boxes": [
                [75, 275, 1725, 850],
                [425, 600, 700, 875],
                [1375, 550, 1650, 800],
                [1240, 675, 1400, 750],
            ]
        },
        {
            "prompt_type":["everything"]
        },
    ]
    
    init_time = time.time()
    segmenter = BaseSegmenter(
        device='cuda',
        # checkpoint='sam_vit_h_4b8939.pth',
        checkpoint='segmenter/sam_vit_h_4b8939.pth',
        model_type='vit_h'
    )
    print(f'init time: {time.time() - init_time}')
    
    infer_time = time.time()
    for i, prompt in enumerate(prompts):
        print(f'{prompt["prompt_type"]} mode')
        masks = segmenter.inference(image_path, prompt)
        print(masks.shape)
        
    print(f'infer time: {time.time() - infer_time}')
