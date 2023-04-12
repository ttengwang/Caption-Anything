# Caption-Anything
<!-- ![](./Image/title.svg) -->
**Caption-Anything** is a versatile image processing tool that combines the capabilities of [Segment Anything](https://github.com/facebookresearch/segment-anything), Visual Captioning, and [ChatGPT](https://openai.com/blog/chatgpt). Our solution generates descriptive captions for any object within an image, offering a range of language styles to accommodate diverse user preferences. **Caption-Anything** supports visual controls (mouse click) and language controls (length, sentiment, factuality, and language).
* visual controls and language controls for text generation
* Chat about selected object for detailed understanding
* Interactive demo  
<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" href="https://huggingface.co/spaces/TencentARC/Caption-Anything">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" alt="Open in Spaces">
</a>

![](./Image/UI.png)


<!-- <a src="https://colab.research.google.com/assets/colab-badge.svg" href="">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a> -->

### Demo
Explore the interactive demo of Caption-Anything, which showcases its powerful capabilities in generating captions for various objects within an image. The demo allows users to control visual aspects by clicking on objects, as well as to adjust textual properties such as length, sentiment, factuality, and language.
![](./Image/demo1.png)

---

![](./Image/demo2.png)

### Getting Started


* Clone the repository:
```bash
git clone https://github.com/ttengwang/caption-anything.git
```
* Install dependencies:
```bash
cd caption-anything
pip install -r requirements.txt
```
* Download the [SAM checkpoints](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place it to `./segmenter/sam_vit_h_4b8939.pth.`

* Run the Caption-Anything gradio demo.
```bash
# Configure the necessary ChatGPT APIs
export OPENAI_API_KEY={Your_Private_Openai_Key}
python app.py --captioner blip2 --port 6086
```

## Acknowledgement
The project is based on [Segment Anything](https://github.com/facebookresearch/segment-anything), BLIP/BLIP-2, [ChatGPT](https://openai.com/blog/chatgpt). Thanks for the authors for their efforts.
