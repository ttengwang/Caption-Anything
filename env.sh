conda create -n caption_anything python=3.8.8 -y 
source activate caption_anything
pip install -r requirements.txt 
cd segmenter
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth segmenter/sam_vit_h_4b8939.pth