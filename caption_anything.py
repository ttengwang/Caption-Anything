from captioner import build_captioner, BaseCaptioner
from segmenter import build_segmenter
from text_refiner import build_text_refiner
import os
import argparse
import pdb
import time
from PIL import Image

class CaptionAnything():
    def __init__(self, args):
        self.args = args
        self.captioner = build_captioner(args.captioner, args.device, args)
        self.segmenter = build_segmenter(args.segmenter, args.device, args)
        if not args.disable_gpt:
            self.init_refiner()


    def init_refiner(self):
        if os.environ.get('OPENAI_API_KEY', None):
            self.text_refiner = build_text_refiner(self.args.text_refiner, self.args.device, self.args)
            
    def inference(self, image, prompt, controls, disable_gpt=False):
        #  segment with prompt
        print("CA prompt: ", prompt, "CA controls",controls)
        seg_mask = self.segmenter.inference(image, prompt)[0, ...]
        mask_save_path = f'result/mask_{time.time()}.png'
        if not os.path.exists(os.path.dirname(mask_save_path)):
            os.makedirs(os.path.dirname(mask_save_path))
        new_p = Image.fromarray(seg_mask.astype('int') * 255.)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(mask_save_path)
        print('seg_mask path: ', mask_save_path)
        print("seg_mask.shape: ", seg_mask.shape)
        #  captioning with mask
        if self.args.enable_reduce_tokens:
            caption, crop_save_path = self.captioner.inference_with_reduced_tokens(image, seg_mask, crop_mode=self.args.seg_crop_mode, filter=self.args.clip_filter, regular_box = self.args.regular_box)
        else:
            caption, crop_save_path = self.captioner.inference_seg(image, seg_mask, crop_mode=self.args.seg_crop_mode, filter=self.args.clip_filter, regular_box = self.args.regular_box)
        #  refining with TextRefiner
        context_captions = []
        if self.args.context_captions:
            context_captions.append(self.captioner.inference(image))
        if not disable_gpt and hasattr(self, "text_refiner"):
            refined_caption = self.text_refiner.inference(query=caption, controls=controls, context=context_captions)
        else:
            refined_caption = {'raw_caption': caption}                
        out = {'generated_captions': refined_caption,
            'crop_save_path': crop_save_path,
            'mask_save_path': mask_save_path,
            'context_captions': context_captions}
        return out
    
def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captioner', type=str, default="blip")
    parser.add_argument('--segmenter', type=str, default="base")
    parser.add_argument('--text_refiner', type=str, default="base")
    parser.add_argument('--segmenter_checkpoint', type=str, default="segmenter/sam_vit_h_4b8939.pth")
    parser.add_argument('--seg_crop_mode', type=str, default="w_bg", choices=['wo_bg', 'w_bg'], help="whether to add or remove background of the image when captioning")
    parser.add_argument('--clip_filter', action="store_true", help="use clip to filter bad captions")
    parser.add_argument('--context_captions', action="store_true", help="use surrounding captions to enhance current caption (TODO)")
    parser.add_argument('--regular_box', action="store_true", default = False, help="crop image with a regular box")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--port', type=int, default=6086, help="only useful when running gradio applications")  
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--gradio_share', action="store_true")
    parser.add_argument('--disable_gpt', action="store_true")
    parser.add_argument('--enable_reduce_tokens', action="store_true", default=False)
    parser.add_argument('--disable_reuse_features', action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args

if __name__ == "__main__":
    args = parse_augment()
    # image_path = 'test_img/img3.jpg'
    image_path = 'test_img/img13.jpg'
    prompts = [
        {
            "prompt_type":["click"],
            "input_point":[[500, 300], [1000, 500]],
            "input_label":[1, 0],
            "multimask_output":"True",
        },
        {
            "prompt_type":["click"],
            "input_point":[[900, 800]],
            "input_label":[1],
            "multimask_output":"True",
        }
    ]
    controls = {
            "length": "30",
            "sentiment": "positive",
            # "imagination": "True",
            "imagination": "False",
            "language": "English",
        }
    
    model = CaptionAnything(args)
    for prompt in prompts:
        print('*'*30)
        print('Image path: ', image_path)
        image = Image.open(image_path)
        print(image)
        print('Visual controls (SAM prompt):\n', prompt)
        print('Language controls:\n', controls)
        out = model.inference(image_path, prompt, controls)
    
    