from .base_segmenter import BaseSegmenter
from caption_anything.utils.utils import seg_model_map
import copy

def build_segmenter(model_name, device, args, model=None):
        return BaseSegmenter(device, args.segmenter_checkpoint, model_name, reuse_feature=not args.disable_reuse_features, model=model, args=args)

def build_segmenter_densecap(model_name, device, args, model=None):
        args_for_densecap = copy.deepcopy(args)
        args_for_densecap.pred_iou_thresh = 0.88
        args_for_densecap.min_mask_region_area = 400
        args_for_densecap.stability_score_thresh = 0.95
        args_for_densecap.box_nms_thresh = 0.3
        return BaseSegmenter(device, args.segmenter_checkpoint, model_name, reuse_feature=not args.disable_reuse_features, model=model, args=args)