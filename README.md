# Caption-Anything
<!-- ![](./Image/title.svg) -->
**Caption-Anything** is a versatile image processing tool that combines the capabilities of [Segment Anything](https://github.com/facebookresearch/segment-anything), Visual Captioning, and [ChatGPT](https://openai.com/blog/chatgpt). Our solution generates descriptive captions for any object within an image, offering a range of language styles to accommodate diverse user preferences. **Caption-Anything** supports visual controls (mouse click) and language controls (length, sentiment, factuality, and language).
* visual controls and language controls for text generation
* Chat about selected object for detailed understanding
* Interactive demo
![](./Image/UI.png)

<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" href="https://huggingface.co/spaces/wybertwang/Caption-Anything">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" alt="Open in Spaces">
</a>
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
git clone https://github.com/tengwang/caption-anything.git
```
* Install dependencies:
```bash
cd caption-anything
pip install -r requirements.txt
```
* Configure the necessary ChatGPT APIs
```bash
export OPENAI_API_KEY={Your_Private_Openai_Key}
```
* Run the Caption-Anything gradio demo.
```bash
python app.py --regular_box  --port 6086
```

## Acknowledgement
The project is based on [Segment Anything](https://github.com/facebookresearch/segment-anything), BLIP/BLIP-2, [ChatGPT](https://openai.com/blog/chatgpt). Thanks for the authors for their efforts.
