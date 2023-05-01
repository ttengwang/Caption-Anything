import argparse

def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captioner', type=str, default="blip2")
    parser.add_argument('--segmenter', type=str, default="huge")
    parser.add_argument('--text_refiner', type=str, default="base")
    parser.add_argument('--segmenter_checkpoint', type=str, default=None, help="SAM checkpoint path")
    parser.add_argument('--seg_crop_mode', type=str, default="wo_bg", choices=['wo_bg', 'w_bg'],
                        help="whether to add or remove background of the image when captioning")
    parser.add_argument('--clip_filter', action="store_true", help="use clip to filter bad captions")
    parser.add_argument('--context_captions', action="store_true",
                        help="use surrounding captions to enhance current caption (TODO)")
    parser.add_argument('--disable_regular_box', action="store_true", default=False,
                        help="crop image with a regular box")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--port', type=int, default=6086, help="only useful when running gradio applications")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--gradio_share', action="store_true")
    parser.add_argument('--disable_gpt', action="store_true")
    parser.add_argument('--enable_reduce_tokens', action="store_true", default=False)
    parser.add_argument('--disable_reuse_features', action="store_true", default=False)
    parser.add_argument('--enable_morphologyex', action="store_true", default=False)
    parser.add_argument('--chat_tools_dict', type=str, default='VisualQuestionAnswering_cuda:0', help='Visual ChatGPT tools, only useful when running gradio applications')
    
    parser.add_argument('--pred_iou_thresh', type=float, default=0.88, help="sam post-precessing")  
    parser.add_argument('--min_mask_region_area', type=int, default=0, help="sam post-precessing")
    parser.add_argument('--stability_score_thresh', type=float, default=0.95, help='sam post-processing')
    parser.add_argument('--box_nms_thresh', type=float, default=0.7, help='sam post-processing')
    
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args
