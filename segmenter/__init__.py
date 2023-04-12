from segmenter.base_segmenter import BaseSegmenter


def build_segmenter(type, device, args=None, model=None):
    if type == 'base':
        return BaseSegmenter(device, args.segmenter_checkpoint, reuse_feature=not args.disable_reuse_features, model=model)
    else:
        raise NotImplementedError()