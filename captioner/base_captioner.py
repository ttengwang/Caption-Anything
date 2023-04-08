import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import json
import pdb
import cv2
import numpy as np
from typing import Union

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
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def cut_box(img, rect_points):
    dst_pts = np.array([[img.shape[1], img.shape[0]], [0, img.shape[0]], [0, 0], [img.shape[1], 0]], dtype="float32")    
    # dst_pts = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]], dtype="float32")
    transform = cv2.getPerspectiveTransform(rect_points.astype("float32"), dst_pts)
    cropped_img = cv2.warpPerspective(img, transform, (img.shape[1], img.shape[0]))
    return cropped_img
    
class BaseCaptioner:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = None
        self.model = None

    def inference(self, image: Union[np.ndarray, Image.Image, str]):
        raise NotImplementedError()
    
    def inference_box(self, image: Union[np.ndarray, Image.Image, str], box: Union[list, np.ndarray]):
        if type(image) == str: # input path
            image = Image.open(image)
        elif type(image) == np.ndarray:
            image = Image.fromarray(image)

        if np.array(box).size == 4: # [x0, y0, x1, y1], where (x0, y0), (x1, y1) represent top-left and bottom-right corners
            size = max(image.width, image.height)
            x1, y1, x2, y2 = box
            image_crop = image.crop((x1 * size, y1 * size, x2 * size, y2 * size))       
        elif np.array(box).size == 8: # four corners of an irregular rectangle
            image_crop = cut_box(np.array(image), box)

        Image.fromarray(image_crop).save('result/crop.png')
        print('croped image saved in result/crop.png')
        return self.inference(image_crop)

    def inference_seg(self, image: Union[np.ndarray, str], seg_mask: Union[np.ndarray, Image.Image, str]):
        if type(image) == str:
            image = Image.open(image)
        if type(seg_mask) == str:
            seg_mask = Image.open(seg_mask)
        elif type(seg_mask) == np.ndarray:
            seg_mask = Image.fromarray(seg_mask)
        seg_mask = seg_mask.resize(image.size)
        seg_mask = np.array(seg_mask) > 0
        seg_no_background = np.array(image) * seg_mask[:,:,np.newaxis]
        min_area_box = seg_to_box(seg_mask)
        return self.inference_box(seg_no_background, min_area_box)

        
if __name__ == '__main__':
    model = BaseCaptioner(device='cuda:0')
    image_path = 'test_img/img2.jpg'
    seg_mask = np.zeros((15,15))
    seg_mask[5:10, 5:10] = 1
    seg_mask = 'image/SAM/img10.jpg.raw_mask.png'
    print(model.inference_seg(image_path, seg_mask))
    