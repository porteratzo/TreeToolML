source ~/anaconda3/etc/profile.d/conda.sh
conda remove -n treetoolml --all -y
echo Creating environment
conda create -n treetoolml -c conda-forge/label/gcc7 qhull -y
conda activate treetoolml && conda install -c conda-forge -c davidcaron pclpy cudatoolkit-dev -y 
conda activate treetoolml && conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
echo Installing tictoc
pip install setuptools==59.5.0
#conda activate treetoolml && conda install -c conda-forge cudatoolkit-dev -y
conda activate treetoolml && cd /home/omar/Documents/mine/MY_LIBS/tictoc && pip install -e .
conda activate treetoolml && cd /home/omar/Documents/mine/MY_LIBS/Libraries && pip install -e .
conda activate treetoolml && cd /home/omar/Documents/mine/MY_LIBS/TreeTool && pip install -e .
conda activate treetoolml && cd /home/omar/Documents/mine/MY_LIBS/TreeToolML && pip install -e .







