<div align="center">
    <img src="assets/caption_anything_logo.png" height="160" />
</div>
<div align="center">
<!-- <h1 align="center"> Caption Anything </h1> -->
<a src="https://img.shields.io/badge/arXiv-2305.02677-b31b1b.svg" href="https://arxiv.org/abs/2305.02677">
    <img src="https://img.shields.io/badge/arXiv-2305.02677-b31b1b.svg">
</a>
<a src="https://colab.research.google.com/assets/colab-badge.svg" href="https://colab.research.google.com/github/ttengwang/Caption-Anything/blob/main/notebooks/tutorial.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>
<a src="https://img.shields.io/badge/%F0%9F%9A%80-SUSTech_VIP_Lab-important.svg" href="https://zhengfenglab.com/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-SUSTech_VIP_Lab-important.svg">
</a>
</div>

[***Caption-Anything***](https://arxiv.org/abs/2305.02677) is a versatile image processing tool that combines the capabilities of [Segment Anything](https://github.com/facebookresearch/segment-anything), Visual Captioning, and [ChatGPT](https://openai.com/blog/chatgpt). Our solution generates descriptive captions for any object within an image, offering a range of language styles to accommodate diverse user preferences. It supports visual controls (mouse click) and language controls (length, sentiment, factuality, and language).
* Visual controls and language controls for text generation
* Chat about selected object for detailed understanding
* Interactive demo

<div align=center>
<img src="./assets/qingming.gif" />
<br>    
Along the River During the Qingming Festival (清明上河图)
</div>
<br> 

### :rocket: Updates
* 2023/04/30: support caption everything in a paragraph
* 2023/04/25: We are delighted to introduce [Track-Anything](https://github.com/gaomingqi/Track-Anything), an inventive project from our lab that offers a versatile and user-friendly solution for video object tracking and segmentation.
* 2023/04/23: support langchain + VQA, better chatbox performance
* 2023/04/20: add mouse trajectory as visual control (beta)
* 2023/04/13: add Colab Tutorial <a src="https://colab.research.google.com/assets/colab-badge.svg" href="https://colab.research.google.com/github/ttengwang/Caption-Anything/blob/main/notebooks/tutorial.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"> </a>
* 2023/04/11: Release code

### :joystick: Demo
Explore the interactive demo of Caption-Anything, which showcases its powerful capabilities in generating captions for various objects within an image. The demo allows users to control visual aspects by clicking on objects, as well as to adjust textual properties such as length, sentiment, factuality, and language.

---

![](./assets/UI.png)

---

![](./assets/demo1.png)

---

![](./assets/demo2.png)

### :hammer_and_wrench: Getting Started

#### Linux
```bash
# Clone the repository:
git clone https://github.com/ttengwang/caption-anything.git
cd caption-anything

# Install dependencies (python version >= 3.8.1):
pip install -r requirements.txt

# Configure the necessary ChatGPT APIs
export OPENAI_API_KEY={Your_Private_Openai_Key}

# Run the Caption-Anything gradio demo.
python app_langchain.py --segmenter huge --captioner blip2 --port 6086  --clip_filter  # requires 13G GPU memory
#python app_langchain.py --segmenter base --captioner blip2 # requires 8.5G GPU memory
#python app_langchain.py --segmenter base --captioner blip # requires 5.5G GPU memory

# (Optional) Use the pre-downloaded SAM checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth ./sam_vit_h_4b8939.pth
python app_langchain.py --segmenter huge --captioner blip2 --port 6086 --segmenter_checkpoint ./sam_vit_b_01ec64.pth  # requires 11.7G GPU memory
```

#### Windows(powershell)
Tested in Windows11 using Nvidia 3070-8G.

```shell
# Clone the repository:
git clone https://github.com/ttengwang/caption-anything.git
cd caption-anything

# Install dependencies:
pip install -r requirements.txt

# Download the [base SAM checkpoints](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).
Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile ./sam_vit_b_01ec64.pth

# Configure the necessary ChatGPT APIs
$env:OPENAI_API_KEY = '{Your_Private_Openai_Key}'

# Run the Caption-Anything gradio demo.
python app_langchain.py --captioner blip --port 6086 --segmenter base # better chatbox via langchain + VQA
python app_langchain.py --captioner blip --port 6086 --segmenter base --segmenter_checkpoint ./sam_vit_b_01ec64.pth  # Use the pre-downloaded SAM checkpoints
python app.py --captioner blip --port 6086 --segmenter base 
```

## :computer: Usage
```python
from caption_anything import CaptionAnything, parse_augment
args = parse_augment()
visual_controls = {
    "prompt_type":["click"],
    "input_point":[[500, 300], [1000, 500]],
    "input_label":[1, 0], # 1/0 for positive/negative points
    "multimask_output":"True",
}
language_controls = {
    "length": "30",
    "sentiment": "natural", # "positive","negative", "natural"
    "imagination": "False", # "True", "False"
    "language": "English" # "Chinese", "Spanish", etc.
}
model = CaptionAnything(args, openai_api_key)
out = model.inference(image_path, visual_controls, language_controls)
```
## :book: Citation
If you find this work useful for your research, please cite our github repo:

```bibtex
@article{wang2023caption,
  title={Caption anything: Interactive image description with diverse multimodal controls},
  author={Wang, Teng and Zhang, Jinrui and Fei, Junjie and Ge, Yixiao and Zheng, Hao and Tang, Yunlong and Li, Zhe and Gao, Mingqi and Zhao, Shanshan and Shan, Ying and Zheng, Feng},
  journal={arXiv preprint arXiv:2305.02677},
  year={2023}
}
```
## Acknowledgements
The project is based on [Segment Anything](https://github.com/facebookresearch/segment-anything), [BLIP/BLIP-2](https://github.com/salesforce/LAVIS), [ChatGPT](https://openai.com/blog/chatgpt), [Visual ChatGPT](https://github.com/microsoft/TaskMatrix), [GiT](https://github.com/microsoft/GenerativeImage2Text). Thanks for the authors for their efforts.
## Contributor
Our project wouldn't be possible without the contributions of these amazing people! Thank you all for making this project better.

[Teng Wang](http://ttengwang.com/) @ Southern University of Science and Technology & HKU & Tencent ARC Lab \
[Jinrui Zhang](https://github.com/zjr2000) @ Southern University of Science and Technology \
[Junjie Fei](https://github.com/JunjieFei) @ Xiamen University \
[Zhe Li](https://github.com/memoryunreal) @ Southern University of Science and Technology \
[Yunlong Tang](https://github.com/yunlong10) @ Southern University of Science and Technology \
[Mingqi Gao](https://mingqigao.com/) @ Southern University of Science and Technology & University of Warwick \
[Hao Zheng](https://github.com/zh-plus) @ Southern University of Science and Technology

<a href="https://github.com/ttengwang/Caption-Anything/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ttengwang/Caption-Anything" />
</a>
