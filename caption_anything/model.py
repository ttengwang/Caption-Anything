import os
import argparse
import pdb
import time
from PIL import Image
import cv2
import numpy as np
from caption_anything.captioner import build_captioner, BaseCaptioner
from caption_anything.segmenter import build_segmenter
from caption_anything.text_refiner import build_text_refiner


class CaptionAnything:
    def __init__(self, args, api_key="", captioner=None, segmenter=None, text_refiner=None):
        self.args = args
        self.captioner = build_captioner(args.captioner, args.device, args) if captioner is None else captioner
        self.segmenter = build_segmenter(args.segmenter, args.device, args) if segmenter is None else segmenter

        self.text_refiner = None
        if not args.disable_gpt:
            if text_refiner is not None:
                self.text_refiner = text_refiner
            else:
                self.init_refiner(api_key)

    @property
    def image_embedding(self):
        return self.segmenter.image_embedding

    @image_embedding.setter
    def image_embedding(self, image_embedding):
        self.segmenter.image_embedding = image_embedding

    @property
    def original_size(self):
        return self.segmenter.predictor.original_size

    @original_size.setter
    def original_size(self, original_size):
        self.segmenter.predictor.original_size = original_size

    @property
    def input_size(self):
        return self.segmenter.predictor.input_size

    @input_size.setter
    def input_size(self, input_size):
        self.segmenter.predictor.input_size = input_size

    def setup(self, image_embedding, original_size, input_size, is_image_set):
        self.image_embedding = image_embedding
        self.original_size = original_size
        self.input_size = input_size
        self.segmenter.predictor.is_image_set = is_image_set

    def init_refiner(self, api_key):
        try:
            self.text_refiner = build_text_refiner(self.args.text_refiner, self.args.device, self.args, api_key)
            self.text_refiner.llm('hi')  # test
        except:
            self.text_refiner = None
            print('OpenAI GPT is not available')

    def inference(self, image, prompt, controls, disable_gpt=False, enable_wiki=False):
        #  segment with prompt
        print("CA prompt: ", prompt, "CA controls", controls)
        seg_mask = self.segmenter.inference(image, prompt)[0, ...]
        if self.args.enable_morphologyex:
            seg_mask = 255 * seg_mask.astype(np.uint8)
            seg_mask = np.stack([seg_mask, seg_mask, seg_mask], axis=-1)
            seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel=np.ones((6, 6), np.uint8))
            seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel=np.ones((6, 6), np.uint8))
            seg_mask = seg_mask[:, :, 0] > 0
        mask_save_path = f'result/mask_{time.time()}.png'
        if not os.path.exists(os.path.dirname(mask_save_path)):
            os.makedirs(os.path.dirname(mask_save_path))
        seg_mask_img = Image.fromarray(seg_mask.astype('int') * 255.)
        if seg_mask_img.mode != 'RGB':
            seg_mask_img = seg_mask_img.convert('RGB')
        seg_mask_img.save(mask_save_path)
        print('seg_mask path: ', mask_save_path)
        print("seg_mask.shape: ", seg_mask.shape)
        #  captioning with mask
        if self.args.enable_reduce_tokens:
            caption, crop_save_path = self.captioner. \
                inference_with_reduced_tokens(image, seg_mask,
                                              crop_mode=self.args.seg_crop_mode,
                                              filter=self.args.clip_filter,
                                              disable_regular_box=self.args.disable_regular_box)
        else:
            caption, crop_save_path = self.captioner. \
                inference_seg(image, seg_mask, crop_mode=self.args.seg_crop_mode,
                              filter=self.args.clip_filter,
                              disable_regular_box=self.args.disable_regular_box)
        #  refining with TextRefiner
        context_captions = []
        if self.args.context_captions:
            context_captions.append(self.captioner.inference(image))
        if not disable_gpt and self.text_refiner is not None:
            refined_caption = self.text_refiner.inference(query=caption, controls=controls, context=context_captions,
                                                          enable_wiki=enable_wiki)
        else:
            refined_caption = {'raw_caption': caption}
        out = {'generated_captions': refined_caption,
               'crop_save_path': crop_save_path,
               'mask_save_path': mask_save_path,
               'mask': seg_mask_img,
               'context_captions': context_captions}
        return out


if __name__ == "__main__":
    from caption_anything.utils.parser import parse_augment
    args = parse_augment()
    # image_path = 'test_images/img3.jpg'
    image_path = 'test_images/img1.jpg'
    prompts = [
        {
            "prompt_type": ["click"],
            "input_point": [[500, 300], [200, 500]],
            "input_label": [1, 0],
            "multimask_output": "True",
        },
        {
            "prompt_type": ["click"],
            "input_point": [[300, 800]],
            "input_label": [1],
            "multimask_output": "True",
        }
    ]
    controls = {
        "length": "30",
        "sentiment": "positive",
        # "imagination": "True",
        "imagination": "False",
        "language": "English",
    }

    model = CaptionAnything(args, os.environ['OPENAI_API_KEY'])
    for prompt in prompts:
        print('*' * 30)
        print('Image path: ', image_path)
        image = Image.open(image_path)
        print(image)
        print('Visual controls (SAM prompt):\n', prompt)
        print('Language controls:\n', controls)
        out = model.inference(image_path, prompt, controls)
