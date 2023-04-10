conda create -n caption_anything python=3.9 -y && conda activate caption_anything
pip install -r requirement.txt 
cd segmenter
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 

