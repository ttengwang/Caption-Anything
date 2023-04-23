from .base_segmenter import BaseSegmenter
from caption_anything.utils.utils import seg_model_map

def build_segmenter(model_name, device, args=None, model=None):
        return BaseSegmenter(device, args.segmenter_checkpoint, model_name, reuse_feature=not args.disable_reuse_features, model=model)