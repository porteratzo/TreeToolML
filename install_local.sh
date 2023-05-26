source ~/anaconda3/etc/profile.d/conda.sh
echo Creating environment
conda env create
echo Installing tictoc
conda activate treetoolml && cd /home/omar/Documents/mine/MY_LIBS/tictoc && pip install -e .
conda activate treetoolml && cd /home/omar/Documents/mine/MY_LIBS/Libraries && pip install -e .
conda activate treetoolml && cd /home/omar/Documents/mine/MY_LIBS/TreeTool && pip install -e .
conda activate treetoolml && cd /home/omar/Documents/mine/MY_LIBS/TreeToolML && pip install -e .






