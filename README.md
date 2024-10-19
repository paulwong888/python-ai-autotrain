# python-ai-autotrain

### install env
    conda create -n autotrain python=3.10
    conda activate autotrain
    pip install autotrain-advanced -i https://pypi.mirrors.ustc.edu.cn/simple/
    conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
    conda install -c "nvidia/label/cuda-12.1.0" cuda-nvcc
    conda install xformers -c xformers
    python -m nltk.downloader punkt
    pip install flash-attn --no-build-isolation # if you want to use flash-attn
    pip install deepspeed # if you want to use deepspeed

### anaconda
    conda env export > environment.yml
    conda config --set custom_channels.auto https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/