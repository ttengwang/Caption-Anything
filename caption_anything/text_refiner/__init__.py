from .text_refiner import TextRefiner


def build_text_refiner(type, device, args=None, api_key=""):
    if type == 'base':
        return TextRefiner(device, api_key)