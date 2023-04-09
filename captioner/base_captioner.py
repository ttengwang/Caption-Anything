import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import json
import pdb
import cv2
import numpy as np
from typing import Union
import time
import clip

def boundary(inputs):
    
    col = inputs.shape[1]
    inputs = inputs.reshape(-1)
    lens = len(inputs)

    for i in range(lens):
        if inputs[i] != False:
            break
    for j in range(lens):
        if inputs[lens - 1 - j] != False:
            break
    start = i
    end = lens - 1 - j
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
        assert seg_mask.ndim == 2 # only support single-channel segmentation mask
        seg_mask = seg_mask.astype('uint8')
        if seg_mask.dtype == 'bool':
            seg_mask = seg_mask * 255
    contours, hierarchy = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.concatenate(contours, axis=0)
    rect = cv2.minAreaRect(contours)
    box = cv2.boxPoints(rect)
    if rect[-1] >= 45:
        newstart = box.argmin(axis=0)[1] # leftmost
    else:
        newstart = box.argmax(axis=0)[0] # topmost
    box = np.concatenate([box[newstart:], box[:newstart]], axis=0)
    box = np.int0(box)
    return box

def get_w_h(rect_points):
    w = np.linalg.norm(rect_points[0] - rect_points[1], ord=2).astype('int')
    h = np.linalg.norm(rect_points[0] - rect_points[3], ord=2).astype('int')
    return w, h
    
def cut_box(img, rect_points):
    w, h = get_w_h(rect_points)
    dst_pts = np.array([[h, 0], [h, w], [0, w], [0, 0],], dtype="float32")
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
        self.threshold = 0.2

    @torch.no_grad()
    def filter_caption(self, image: Union[np.ndarray, Image.Image, str], caption: str):
 
        if type(image) == str: # input path
            image = Image.open(image)
        elif type(image) == np.ndarray:
            image = Image.fromarray(image)
 
        image = self.preprocess(image).unsqueeze(0).to(self.device) # (1, 3, 224, 224)
        text = clip.tokenize(caption).to(self.device)               # (1, 77)
        image_features = self.filter.encode_image(image) # (1, 512)
        text_features = self.filter.encode_text(text)    # (1, 512)
        image_features /= image_features.norm(dim = -1, keepdim = True)
        text_features /= text_features.norm(dim = -1, keepdim = True)
        similarity = torch.matmul(image_features, text_features.transpose(1, 0)).item()
        if similarity < self.threshold:
            print('There seems to be nothing where you clicked.')
            out = ""
        else:
            out = caption
        print(f'Clip score of the caption is {similarity}')
        return out

        
    def inference(self, image: Union[np.ndarray, Image.Image, str], filter: bool=False):
        raise NotImplementedError()
    
    def inference_with_reduced_tokens(self, image: Union[np.ndarray, Image.Image, str], seg_mask):
        raise NotImplementedError()
    
    def inference_box(self, image: Union[np.ndarray, Image.Image, str], box: Union[list, np.ndarray], filter=False):
        if type(image) == str: # input path
            image = Image.open(image)
        elif type(image) == np.ndarray:
            image = Image.fromarray(image)

        if np.array(box).size == 4: # [x0, y0, x1, y1], where (x0, y0), (x1, y1) represent top-left and bottom-right corners
            size = max(image.width, image.height)
            x1, y1, x2, y2 = box
            image_crop = np.array(image.crop((x1 * size, y1 * size, x2 * size, y2 * size)))  
        elif np.array(box).size == 8: # four corners of an irregular rectangle
            image_crop = cut_box(np.array(image), box)

        crop_save_path = f'result/crop_{time.time()}.png'
        Image.fromarray(image_crop).save(crop_save_path)
        print(f'croped image saved in {crop_save_path}')
        caption = self.inference(image_crop, filter)
        return caption, crop_save_path
        

    def inference_seg(self, image: Union[np.ndarray, str], seg_mask: Union[np.ndarray, Image.Image, str], crop_mode="w_bg", filter=False):
        if type(image) == str:
            image = Image.open(image)
        if type(seg_mask) == str:
            seg_mask = Image.open(seg_mask)
        elif type(seg_mask) == np.ndarray:
            seg_mask = Image.fromarray(seg_mask)
        seg_mask = seg_mask.resize(image.size)
        seg_mask = np.array(seg_mask) > 0
        
        if crop_mode=="wo_bg":
            image = np.array(image) * seg_mask[:,:,np.newaxis]
        else:
            image = np.array(image)
            
        min_area_box = seg_to_box(seg_mask)
        return self.inference_box(image, min_area_box, filter)
        


        
if __name__ == '__main__':
    model = BaseCaptioner(device='cuda:0')
    image_path = 'test_img/img2.jpg'
    seg_mask = np.zeros((15,15))
    seg_mask[5:10, 5:10] = 1
    seg_mask = 'image/SAM/img10.jpg.raw_mask.png'
    print(model.inference_seg(image_path, seg_mask))
    