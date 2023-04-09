from segmenter.base_segmenter import BaseSegmenter


def build_segmenter(type, device, args=None):
    if type == 'base':
        return BaseSegmenter(device, args.segmenter_checkpoint)