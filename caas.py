from captioner import build_captioner, BaseCaptioner
from segmenter.base_segmenter import BaseSegmenter
from text_refiner.text_refiner import TextRefiner
import os
os.environ['']

class CaptionAnything():
    def __init__(self, 
        captioner: BaseCaptioner,
        segmenter: BaseSegmenter,
        text_refiner: TextRefiner):
        self.captioner = captioner
        self.segmenter = segmenter
        self.text_refiner = text_refiner


    def __call__(self, image_path, prompt, controls):
        # TODO segment with prompt
        seg_mask = self.segmenter.inference(image_path, prompt)
        # TODO captioning with mask
        caption = self.captioner.inference_seg(image_path, seg_mask)
        # TODO refine with TextRefiner
        refined_caption = self.text_refiner.inference(query=caption, controls=controls)

