conda create -n caption_anything python=3.9 -y && conda activate caption_anything
pip install -r requirement.txt 
export OPENAI_API_KEY={Your_Private_Openai_Key}
export OPENAI_API_BASE=https://openai.1rmb.tk/v1/
cd segmenter
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 

