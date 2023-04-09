from text_refiner.text_refiner import TextRefiner


def build_text_refiner(type, device, args=None):
    if type == 'base':
        return TextRefiner(device)