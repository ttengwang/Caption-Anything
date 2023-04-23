from .base_segmenter import BaseSegmenter


def build_segmenter(type, device, args=None, model=None):
        return BaseSegmenter(device, args.segmenter_checkpoint, reuse_feature=not args.disable_reuse_features, model=model)