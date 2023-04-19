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
    def __init__(self, device, checkpoint, model_type='vit_h', reuse_feature=True, model=None):
        print(f"Initializing BaseSegmenter to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = None
        self.model_type = model_type
        if model is None:
            self.checkpoint = checkpoint
            self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
            self.model.to(device=self.device)
        else:
            self.model = model
        self.reuse_feature = reuse_feature
        self.predictor = SamPredictor(self.model)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        self.image_embedding = None
        self.image = None

    def read_image(self, image: Union[np.ndarray, Image.Image, str]):
        if type(image) == str:  # input path
            image = Image.open(image)
            image = np.array(image)
        elif type(image) == Image.Image:
            image = np.array(image)
        elif type(image) == np.ndarray:
            image = image
        else:
            raise TypeError
        return image

    @torch.no_grad()
    def set_image(self, image: Union[np.ndarray, Image.Image, str]):
        image = self.read_image(image)
        self.image = image
        if self.reuse_feature:
            self.predictor.set_image(image)
            self.image_embedding = self.predictor.get_image_embedding()
            print(self.image_embedding.shape)

    @torch.no_grad()
    def inference(self, image: Union[np.ndarray, Image.Image, str], control: dict):
        """
        SAM inference of image according to control.
        Args:
            image: str or PIL.Image or np.ndarray
            control:
                prompt_type:
                    1. {control['prompt_type'] = ['everything']} to segment everything in the image.
                    2. {control['prompt_type'] = ['click', 'box']} to segment according to click and box.
                    3. {control['prompt_type'] = ['click'] to segment according to click.
                    4. {control['prompt_type'] = ['box'] to segment according to box.
                input_point: list of [x, y] coordinates of click.
                input_label: List of labels for points accordingly, 0 for negative, 1 for positive.
                input_box: List of [x1, y1, x2, y2] coordinates of box.
                multimask_output:
                    If true, the model will return three masks.
                    For ambiguous input prompts (such as a single click), this will often
                    produce better masks than a single prediction. If only a single
                    mask is needed, the model's predicted quality score can be used
                    to select the best mask. For non-ambiguous prompts, such as multiple
                    input prompts, multimask_output=False can give better results.
        Returns:
            masks: np.ndarray of shape [num_masks, height, width]

        """
        image = self.read_image(image)  # Turn image into np.ndarray
        if 'everything' in control['prompt_type']:
            masks = self.mask_generator.generate(image)
            new_masks = np.concatenate([mask["segmentation"][np.newaxis, :] for mask in masks])
            return new_masks
        else:
            if not self.reuse_feature or self.image_embedding is None:
                self.set_image(image)
                self.predictor.set_image(self.image)
            else:
                assert self.image_embedding is not None
                self.predictor.features = self.image_embedding

        if 'mutimask_output' in control:
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(control['input_point']),
                point_labels=np.array(control['input_label']),
                multimask_output=True,
            )
        elif 'input_boxes' in control:
            transformed_boxes = self.predictor.transform.apply_boxes_torch(
                torch.tensor(control["input_boxes"], device=self.predictor.device),
                image.shape[1::-1]  # Reverse shape because numpy is (W, H) and function need (H, W)
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
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
            )

            if 0 in control['input_label']:
                mask_input = logits[np.argmax(scores), :, :]
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    mask_input=mask_input[None, :, :],
                    multimask_output=False,
                )

        return masks


if __name__ == "__main__":
    image_path = 'segmenter/images/truck.jpg'
    prompts = [
        # {
        #     "prompt_type":["click"],
        #     "input_point":[[500, 375]],
        #     "input_label":[1],
        #     "multimask_output":"True",
        # },
        {
            "prompt_type": ["click"],
            "input_point": [[1000, 600], [1325, 625]],
            "input_label": [1, 0],
        },
        # {
        #     "prompt_type":["click", "box"],
        #     "input_box":[425, 600, 700, 875],
        #     "input_point":[[575, 750]],
        #     "input_label": [0]
        # },
        # {
        #     "prompt_type":["box"],
        #     "input_boxes": [
        #         [75, 275, 1725, 850],
        #         [425, 600, 700, 875],
        #         [1375, 550, 1650, 800],
        #         [1240, 675, 1400, 750],
        #     ]
        # },
        # {
        #     "prompt_type":["everything"]
        # },
    ]

    init_time = time.time()
    segmenter = BaseSegmenter(
        device='cuda',
        # checkpoint='sam_vit_h_4b8939.pth',
        checkpoint='segmenter/sam_vit_h_4b8939.pth',
        model_type='vit_h',
        reuse_feature=True
    )
    print(f'init time: {time.time() - init_time}')

    image_path = 'test_img/img2.jpg'
    infer_time = time.time()
    for i, prompt in enumerate(prompts):
        print(f'{prompt["prompt_type"]} mode')
        image = Image.open(image_path)
        segmenter.set_image(np.array(image))
        masks = segmenter.inference(np.array(image), prompt)
        Image.fromarray(masks[0]).save('seg.png')
        print(masks.shape)

    print(f'infer time: {time.time() - infer_time}')
