# Installation


The code was tested on windows 10, with [Anaconda](https://www.anaconda.com/download) Python 3.10.16 and [PyTorch]((http://pytorch.org/)) v2.4.1. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:

0. [Optional but recommended] create a new conda environment. 


1. Install pytorch 2.4.1 and torchvision 0.19.1:

 
     
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

3. Clone this repo:


    ~~~
    download the whole MLBFM file.
    ~~~


4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
    
5. Compile multiscale deformable attention 

    ~~~
    cd $MLBFM_ROOT/src/lib/models/networks/ops
    python setup.py build develop
    ~~~
5. Compile lib for polygon regression

    ~~~
    cd $MLBFM_ROOT/src/lib/models/dance_lib/csrc/extreme_utils
    python setup.py build develop
    ~~~

