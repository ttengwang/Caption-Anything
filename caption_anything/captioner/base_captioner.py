import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import json
import pdb
import cv2
import numpy as np
from typing import Any, Union, List
import time
import clip

from caption_anything.utils.utils import load_image


def boundary(inputs):
    col = inputs.shape[1]
    inputs = inputs.reshape(-1)
    lens = len(inputs)
    start = np.argmax(inputs)
    end = lens - 1 - np.argmax(np.flip(inputs))
    top = start // col
    bottom = end // col
    return top, bottom


def new_seg_to_box(seg_mask: Union[np.ndarray, Image.Image, str]):
    if type(seg_mask) == str:
        seg_mask = Image.open(seg_mask)
    elif type(seg_mask) == np.ndarray:
        seg_mask = Image.fromarray(seg_mask)
    seg_mask = np.array(seg_mask) > 0
    size = max(seg_mask.shape[0], seg_mask.shape[1])
    top, bottom = boundary(seg_mask)
    left, right = boundary(seg_mask.T)
    return [left / size, top / size, right / size, bottom / size]


def seg_to_box(seg_mask: Union[np.ndarray, Image.Image, str]):
    if type(seg_mask) == str:
        seg_mask = cv2.imread(seg_mask, cv2.IMREAD_GRAYSCALE)
        _, seg_mask = cv2.threshold(seg_mask, 127, 255, 0)
    elif type(seg_mask) == np.ndarray:
        assert seg_mask.ndim == 2  # only support single-channel segmentation mask
        seg_mask = seg_mask.astype('uint8')
        if seg_mask.dtype == 'bool':
            seg_mask = seg_mask * 255
    contours, hierarchy = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.concatenate(contours, axis=0)
    rect = cv2.minAreaRect(contours)
    box = cv2.boxPoints(rect)
    if rect[-1] >= 45:
        newstart = box.argmin(axis=0)[1]  # leftmost
    else:
        newstart = box.argmax(axis=0)[0]  # topmost
    box = np.concatenate([box[newstart:], box[:newstart]], axis=0)
    box = np.int0(box)
    return box


def get_w_h(rect_points):
    w = np.linalg.norm(rect_points[0] - rect_points[1], ord=2).astype('int')
    h = np.linalg.norm(rect_points[0] - rect_points[3], ord=2).astype('int')
    return w, h


def cut_box(img, rect_points):
    w, h = get_w_h(rect_points)
    dst_pts = np.array([[h, 0], [h, w], [0, w], [0, 0], ], dtype="float32")
    transform = cv2.getPerspectiveTransform(rect_points.astype("float32"), dst_pts)
    cropped_img = cv2.warpPerspective(img, transform, (h, w))
    return cropped_img


class BaseCaptioner:
    def __init__(self, device, enable_filter=False):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = None
        self.model = None
        self.enable_filter = enable_filter
        if enable_filter:
            self.filter, self.preprocess = clip.load('ViT-B/32', device)

    @torch.no_grad()
    def filter_caption(self, image: Union[np.ndarray, Image.Image, str], caption: str, reference_caption: List[str]=[]):
        image = load_image(image, return_type='pil')
        image = self.preprocess(image).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        captions = [caption]
        if len(reference_caption):
            captions.extend(reference_caption)
        text = clip.tokenize(captions).to(self.device)  # (>1, 77)
        image_features = self.filter.encode_image(image)  # (1, 512)
        text_features = self.filter.encode_text(text) # # (>1, 512)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        if len(reference_caption):
            similarity = torch.matmul(image_features, text_features.transpose(1, 0)) / 0.07
            similarity = similarity.softmax(dim=1)[0, 0].item()
        else:
            similarity = torch.matmul(image_features, text_features.transpose(1, 0)).item()
        print(f'Clip score of the caption is {similarity}')
        return similarity

    def inference(self, image: Union[np.ndarray, Image.Image, str], filter: bool = False):
        raise NotImplementedError()

    def inference_with_reduced_tokens(self, image: Union[np.ndarray, Image.Image, str], seg_mask, filter: bool = False):
        raise NotImplementedError()

    def inference_box(self, image: Union[np.ndarray, Image.Image, str], box: Union[list, np.ndarray], filter=False, verbose=False, caption_args={}):
        image = load_image(image, return_type="pil")

        if np.array(box).size == 4:
            # [x0, y0, x1, y1], where (x0, y0), (x1, y1) represent top-left and bottom-right corners
            size = max(image.width, image.height)
            x1, y1, x2, y2 = box
            image_crop = np.array(image.crop((x1 * size, y1 * size, x2 * size, y2 * size)))
        elif np.array(box).size == 8:  # four corners of an irregular rectangle
            image_crop = cut_box(np.array(image), box)

        crop_save_path = None
        if verbose:
            crop_save_path = f'result/crop_{time.time()}.png'
            Image.fromarray(image_crop).save(crop_save_path)
            print(f'croped image saved in {crop_save_path}')
        caption = self.inference(image_crop, filter, caption_args)
        caption.update({'crop_save_path': crop_save_path})
        return caption

    def inference_seg(self, 
                      image: Union[np.ndarray, str], 
                      seg_mask: Union[np.ndarray, Image.Image, str] = None,
                      crop_mode="w_bg", 
                      filter=False, 
                      disable_regular_box=False, 
                      verbose=False, 
                      caption_args={}):
        if seg_mask is None:
            seg_mask = np.ones(image.size).astype(bool)
        
        image = load_image(image, return_type="pil")
        seg_mask = load_image(seg_mask, return_type="pil")

        seg_mask = seg_mask.resize(image.size)
        seg_mask = np.array(seg_mask) > 0
        if crop_mode == "wo_bg":
            image = np.array(image) * seg_mask[:, :, np.newaxis] + (1 - seg_mask[:, :, np.newaxis]) * 255
            image = np.uint8(image)
        else:
            image = np.array(image)

        if disable_regular_box:
            min_area_box = seg_to_box(seg_mask)
        else:
            min_area_box = new_seg_to_box(seg_mask)
        return self.inference_box(image, min_area_box, filter, verbose, caption_args)

    def generate_seg_cropped_image(self, 
                                   image: Union[np.ndarray, str], 
                                   seg_mask: Union[np.ndarray, Image.Image, str],
                                   crop_mode="w_bg", 
                                   disable_regular_box=False):
        image = load_image(image, return_type="pil")
        seg_mask = load_image(seg_mask, return_type="pil")

        seg_mask = seg_mask.resize(image.size)
        seg_mask = np.array(seg_mask) > 0

        if crop_mode == "wo_bg":
            image = np.array(image) * seg_mask[:, :, np.newaxis] + (1 - seg_mask[:, :, np.newaxis]) * 255
        else:
            image = np.array(image)

        if disable_regular_box:
            box = seg_to_box(seg_mask)
        else:
            box = new_seg_to_box(seg_mask)

        if np.array(box).size == 4:
            # [x0, y0, x1, y1], where (x0, y0), (x1, y1) represent top-left and bottom-right corners
            size = max(image.shape[0], image.shape[1])
            x1, y1, x2, y2 = box
            image_crop = np.array(image.crop((x1 * size, y1 * size, x2 * size, y2 * size)))
        elif np.array(box).size == 8:  # four corners of an irregular rectangle
            image_crop = cut_box(np.array(image), box)
        crop_save_path = f'result/crop_{time.time()}.png'
        Image.fromarray(image_crop).save(crop_save_path)
        print(f'croped image saved in {crop_save_path}')
        return crop_save_path


if __name__ == '__main__':
    model = BaseCaptioner(device='cuda:0')
    image_path = 'test_images/img2.jpg'
    seg_mask = np.zeros((15, 15))
    seg_mask[5:10, 5:10] = 1
    seg_mask = 'image/SAM/img10.jpg.raw_mask.png'
    print(model.inference_seg(image_path, seg_mask))
