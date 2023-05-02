import os
import argparse
import pdb
import time
from PIL import Image
import cv2
import numpy as np
from PIL import Image
import easyocr
import copy
import time
from caption_anything.captioner import build_captioner, BaseCaptioner
from caption_anything.segmenter import build_segmenter, build_segmenter_densecap
from caption_anything.text_refiner import build_text_refiner
from caption_anything.utils.utils import prepare_segmenter, seg_model_map, load_image, get_image_shape
from caption_anything.utils.utils import mask_painter_foreground_all, mask_painter, xywh_to_x1y1x2y2, image_resize
from caption_anything.utils.densecap_painter import draw_bbox
            
class CaptionAnything:
    def __init__(self, args, api_key="", captioner=None, segmenter=None, ocr_reader=None, text_refiner=None):
        self.args = args
        self.captioner = build_captioner(args.captioner, args.device, args) if captioner is None else captioner
        self.segmenter = build_segmenter(args.segmenter, args.device, args) if segmenter is None else segmenter
        self.segmenter_densecap = build_segmenter_densecap(args.segmenter, args.device, args, model=self.segmenter.model)
        self.ocr_lang = ["ch_tra", "en"]
        self.ocr_reader = ocr_reader if ocr_reader is not None else easyocr.Reader(self.ocr_lang)
        

        self.text_refiner = None
        if not args.disable_gpt:
            if text_refiner is not None:
                self.text_refiner = text_refiner
            elif api_key != "":
                self.init_refiner(api_key)
        self.require_caption_prompt = args.captioner == 'blip2'
        
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

    def inference(self, image, prompt, controls, disable_gpt=False, enable_wiki=False, verbose=False, is_densecap=False, args={}):
        #  segment with prompt
        print("CA prompt: ", prompt, "CA controls", controls)
        is_seg_everything = 'everything' in prompt['prompt_type']

        args['seg_crop_mode'] = args.get('seg_crop_mode', self.args.seg_crop_mode)
        args['clip_filter'] = args.get('clip_filter', self.args.clip_filter)
        args['disable_regular_box'] = args.get('disable_regular_box', self.args.disable_regular_box)
        args['context_captions'] = args.get('context_captions', self.args.context_captions)
        args['enable_reduce_tokens'] = args.get('enable_reduce_tokens', self.args.enable_reduce_tokens)
        args['enable_morphologyex'] = args.get('enable_morphologyex', self.args.enable_morphologyex)
        args['topN'] = args.get('topN', 10) if is_seg_everything else 1
        args['min_mask_area'] = args.get('min_mask_area', 0)

        if not is_densecap:
            seg_results = self.segmenter.inference(image, prompt)
        else:
            seg_results = self.segmenter_densecap.inference(image, prompt)
            
        seg_masks, seg_bbox, seg_area = seg_results if is_seg_everything else (seg_results, None, None)
        
        if args['topN'] > 1: # sort by area
            samples = list(zip(*[seg_masks, seg_bbox, seg_area]))
            # top_samples = sorted(samples, key=lambda x: x[2], reverse=True)
            # seg_masks, seg_bbox, seg_area = list(zip(*top_samples))
            samples = list(filter(lambda x: x[2] > args['min_mask_area'], samples))
            samples = samples[:args['topN']]
            seg_masks, seg_bbox, seg_area = list(zip(*samples))

        out_list = []
        for i, seg_mask in enumerate(seg_masks):
            if args['enable_morphologyex']:
                seg_mask = 255 * seg_mask.astype(np.uint8)
                seg_mask = np.stack([seg_mask, seg_mask, seg_mask], axis=-1)
                seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel=np.ones((6, 6), np.uint8))
                seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel=np.ones((6, 6), np.uint8))
                seg_mask = seg_mask[:, :, 0] > 0

            seg_mask_img = Image.fromarray(seg_mask.astype('int') * 255.)        
            mask_save_path = None
            
            if verbose:
                mask_save_path = f'result/mask_{time.time()}.png'
                if not os.path.exists(os.path.dirname(mask_save_path)):
                    os.makedirs(os.path.dirname(mask_save_path))

                if seg_mask_img.mode != 'RGB':
                    seg_mask_img = seg_mask_img.convert('RGB')
                seg_mask_img.save(mask_save_path)
                print('seg_mask path: ', mask_save_path)
                print("seg_mask.shape: ", seg_mask.shape)


            #  captioning with mask
            if args['enable_reduce_tokens']:
                result = self.captioner.inference_with_reduced_tokens(image, seg_mask,
                                                  crop_mode=args['seg_crop_mode'],
                                                  filter=args['clip_filter'],
                                                  disable_regular_box=args['disable_regular_box'], 
                                                  verbose=verbose,
                                                  caption_args=args)
            else:
                result = self.captioner.inference_seg(image, seg_mask, 
                                  crop_mode=args['seg_crop_mode'],
                                  filter=args['clip_filter'],
                                  disable_regular_box=args['disable_regular_box'], 
                                  verbose=verbose,
                                  caption_args=args)
            caption = result.get('caption', None)
            crop_save_path = result.get('crop_save_path', None)
            
            #  refining with TextRefiner
            context_captions = []
            if args['context_captions']:
                context_captions.append(self.captioner.inference(image)['caption'])
            if not disable_gpt and self.text_refiner is not None:
                refined_caption = self.text_refiner.inference(query=caption, controls=controls, context=context_captions,
                                                            enable_wiki=enable_wiki)
            else:
                refined_caption = {'raw_caption': caption}
            out = {'generated_captions': refined_caption,
                'crop_save_path': crop_save_path,
                'mask_save_path': mask_save_path,
                'mask': seg_mask_img,
                'bbox': seg_bbox[i] if seg_bbox is not None else None,
                'area': seg_area[i] if seg_area is not None else None,
                'context_captions': context_captions,
                'ppl_score': result.get('ppl_score', -100.),
                'clip_score': result.get('clip_score', 0.)
                }
            out_list.append(out)
        return out_list
    
    def parse_dense_caption(self, image, topN=10, reference_caption=[], verbose=False):
        width, height = get_image_shape(image)
        prompt = {'prompt_type': ['everything']}
        densecap_args = {
            'return_ppl': True, 
            'clip_filter': True, 
            'reference_caption': reference_caption,
            'text_prompt': "", # 'Question: what does the image show? Answer:'
            'seg_crop_mode': 'w_bg',
            # 'text_prompt': "",
            # 'seg_crop_mode': 'wo_bg',
            'disable_regular_box': False,
            'topN': topN,
            'min_ppl_score': -1.8,
            'min_clip_score': 0.30,
            'min_mask_area': 2500,
            }
            
        dense_captions = self.inference(image, prompt,
                                        controls=None, 
                                        disable_gpt=True, 
                                        verbose=verbose, 
                                        is_densecap=True, 
                                        args=densecap_args)
        print('Process Dense Captioning: \n', dense_captions)
        dense_captions = list(filter(lambda x: x['ppl_score'] / (1+len(x['generated_captions']['raw_caption'].split())) >= densecap_args['min_ppl_score'], dense_captions))
        dense_captions = list(filter(lambda x: x['clip_score'] >= densecap_args['min_clip_score'], dense_captions))
        dense_cap_prompt = []
        for cap in dense_captions:
            x, y, w, h = cap['bbox']
            cx, cy = x + w/2, (y + h/2)
            dense_cap_prompt.append("({}: X:{:.0f}, Y:{:.0f}, Width:{:.0f}, Height:{:.0f})".format(cap['generated_captions']['raw_caption'], cx, cy, w, h))
        
        if verbose:
            all_masks = [np.array(item['mask'].convert('P')) for item in dense_captions]
            new_image = mask_painter_foreground_all(np.array(image), all_masks, background_alpha=0.4)
            save_path = 'result/dense_caption_mask.png'
            Image.fromarray(new_image).save(save_path)
            print(f'Dense captioning mask saved in {save_path}')
            
            vis_path = 'result/dense_caption_vis_{}.png'.format(time.time())
            dense_cap_painter_input = [{'bbox': xywh_to_x1y1x2y2(cap['bbox']), 
                                        'caption': cap['generated_captions']['raw_caption']} for cap in dense_captions]
            draw_bbox(load_image(image, return_type='numpy'), vis_path, dense_cap_painter_input, show_caption=True)
            print(f'Dense Captioning visualization saved in {vis_path}')
        return ','.join(dense_cap_prompt)
    
    def parse_ocr(self, image, thres=0.2):
        width, height = get_image_shape(image)
        image = load_image(image, return_type='numpy')
        bounds = self.ocr_reader.readtext(image)
        bounds = [bound for bound in bounds if bound[2] > thres]
        print('Process OCR Text:\n', bounds)
        
        ocr_prompt = []
        for box, text, conf in bounds:
            p0, p1, p2, p3 = box
            ocr_prompt.append('(\"{}\": X:{:.0f}, Y:{:.0f})'.format(text, (p0[0]+p1[0]+p2[0]+p3[0])/4, (p0[1]+p1[1]+p2[1]+p3[1])/4))
        ocr_prompt = '\n'.join(ocr_prompt)
        
        # ocr_prompt = self.text_refiner.llm(f'The image have some scene texts with their locations: {ocr_prompt}. Please group these individual words into one or several phrase based on their relative positions (only give me your answer, do not show explanination)').strip()
        
        # ocr_prefix1 = f'The image have some scene texts with their locations: {ocr_prompt}. Please group these individual words into one or several phrase based on their relative positions (only give me your answer, do not show explanination)'
        # ocr_prefix2 = f'Please group these individual words into 1-3 phrases, given scene texts with their locations: {ocr_prompt}. You return is one or several strings and infer their locations. (only give me your answer like (“man working”, X: value, Y: value), do not show explanination)'
        # ocr_prefix4 = f'summarize the individual scene text words detected by OCR tools into a fluent sentence based on their positions and distances. You should strictly describe all of the given scene text words. Do not miss any given word. Do not create non-exist words. Do not appear numeric positions. The individual words are given:\n{ocr_prompt}\n'
        # ocr_prefix3 = f'combine the individual scene text words detected by OCR tools into one/several fluent phrases/sentences based on their positions and distances. You should strictly copy or correct all of the given scene text words. Do not miss any given word. Do not create non-exist words. The response is several strings seperate with their location (X, Y), each of which represents a phrase. The individual words are given:\n{ocr_prompt}\n'
        # response = self.text_refiner.llm(ocr_prefix3).strip() if len(ocr_prompt) else ""
        return ocr_prompt
    
    def inference_cap_everything(self, image, verbose=False):
        image = load_image(image, return_type='pil')
        image = image_resize(image, res=1024)
        width, height = get_image_shape(image)
        other_args = {'text_prompt': ""} if self.require_caption_prompt else {}
        img_caption = self.captioner.inference(image, filter=False, args=other_args)['caption']
        dense_caption_prompt = self.parse_dense_caption(image, topN=10, verbose=verbose, reference_caption=[])
        scene_text_prompt = self.parse_ocr(image, thres=0.2)
        # scene_text_prompt = "N/A"
        
        # the summarize_prompt is modified from https://github.com/JialianW/GRiT and https://github.com/showlab/Image2Paragraph
        summarize_prompt = "Imagine you are a blind but intelligent image captioner. You should generate a descriptive, coherent and human-like paragraph based on the given information (a,b,c,d) instead of imagination:\na) Image Resolution: {image_size}\nb) Image Caption:{image_caption}\nc) Dense Caption: {dense_caption}\nd) Scene Text: {scene_text}\nThere are some rules for your response: Show objects with their attributes (e.g. position, color, size, shape, texture).\nPrimarily describe common objects with large size.\nProvide context of the image.\nShow relative position between objects.\nLess than 6 sentences.\nDo not appear number.\nDo not describe any individual letter.\nDo not show the image resolution.\nIngore the white background."
        prompt = summarize_prompt.format(**{
            "image_size": "width {} height {}".format(width, height),
            "image_caption":img_caption, 
            "dense_caption": dense_caption_prompt,
            "scene_text": scene_text_prompt})
        print(f'caption everything prompt: {prompt}')
        response = self.text_refiner.llm(prompt).strip()
        # chinese_response = self.text_refiner.llm('Translate it into Chinese: {}'.format(response)).strip()
        return response
        
if __name__ == "__main__":
    from caption_anything.utils.parser import parse_augment
    args = parse_augment()
    image_path = 'result/wt/memes/87226084.jpg'
    image = Image.open(image_path)
    prompts = [
        {
            "prompt_type": ["click"],
            "input_point": [[500, 300], [200, 500]],
            "input_label": [1, 0],
            "multimask_output": "True",
        },
        # {
        #     "prompt_type": ["click"],
        #     "input_point": [[300, 800]],
        #     "input_label": [1],
        #     "multimask_output": "True",
        # }
    ]
    controls = {
        "length": "30",
        "sentiment": "positive",
        # "imagination": "True",
        "imagination": "False",
        "language": "English",
    }

    model = CaptionAnything(args, os.environ['OPENAI_API_KEY'])
    img_dir = 'test_images/memes'
    for image_file in os.listdir(img_dir):
        image_path = os.path.join(img_dir, image_file)  
        print('image_path:', image_path)
        paragraph = model.inference_cap_everything(image_path, verbose=True)
        print('Caption Everything:\n', paragraph)    
        ocr = model.parse_ocr(image_path)
        print('OCR', ocr)