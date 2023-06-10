source ~/anaconda3/etc/profile.d/conda.sh
conda remove -n treetoolml2 --all -y
echo Creating environment
conda env create
echo Installing tictoc
conda activate treetoolml2 && cd /home/omar/Documents/mine/MY_LIBS/tictoc && pip install -e .
conda activate treetoolml2 && cd /home/omar/Documents/mine/MY_LIBS/Libraries && pip install -e .
conda activate treetoolml2 && cd /home/omar/Documents/mine/MY_LIBS/TreeTool && pip install -e .
conda activate treetoolml2 && cd /home/omar/Documents/mine/MY_LIBS/TreeToolML && pip install -e .







