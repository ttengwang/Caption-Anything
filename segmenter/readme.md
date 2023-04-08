### Prepare SAM
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
or
```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```
### Download the checkpoint:

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

### Inference

The prompts are in json format:

```
prompts = [
        {
            "prompt_type":["click"],
            "input_point":[[500, 375]],
            "input_label":[1],
            "multimask_output":"True",
        },
        {
            "prompt_type":["click"],
            "input_point":[[500, 375], [1125, 625]],
            "input_label":[1, 0],
        },
        {
            "prompt_type":["click", "box"],
            "input_box":[425, 600, 700, 875],
            "input_point":[[575, 750]],
            "input_label": [0]
        },
        {
            "prompt_type":["box"],
            "input_boxes": [
                [75, 275, 1725, 850],
                [425, 600, 700, 875],
                [1375, 550, 1650, 800],
                [1240, 675, 1400, 750],
            ]
        },
        {
            "prompt_type":["everything"]
        },
    ]
```

In `base_segmenter.py`:
```
segmenter = BaseSegmenter(
        device='cuda',
        checkpoint='sam_vit_h_4b8939.pth',
        model_type='vit_h'
    )

for i, prompt in enumerate(prompts):
    masks = segmenter.inference(image_path, prompt)
```

Outputs are masks (True and False numpy Matrix), shape: (num of masks, height, weight)