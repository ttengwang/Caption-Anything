conda create -n caption python=3.9 -y 
source activate caption
pip install -r requirements.txt 
cd segmenter
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 

