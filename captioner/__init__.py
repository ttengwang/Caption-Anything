from blip import BLIPCaptioner
from blip2 import BLIP2Captioner
from git import GITCaptioner
from base_captioner import BaseCaptioner


def build_captioner(type, device):
    if type == 'blip':
        return BLIPCaptioner(device)
    elif type == 'blip2':
        return BLIP2Captioner(device)
    elif type == 'git':
        return GITCaptioner(device)
    else:
        raise NotImplementedError("")