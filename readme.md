# Football playing AI agent



## Requirements:
-python >=3.6
-pytorch (tested on version 1.3) [pip install torch torchvision] #Assuming cuda 10.2 on the system(refer pytorch website)
-google research platform

## Steps to install Google Football Research Environment

```
- sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip
-git clone https://github.com/google-research/football.git
-cd football
-pip3 install .

```

# Test the installation of football platform

```
cd into the repo
Run "python environment_test.py"
```

It will continuously render the frames:

## To train the Football playing agent

-Download my repo
-cd into the repo
-Run
```python train.py```

## To test the model

Requirement:A trained model for 6000 epochs is added in the repo in the name "PPO_act6000.pth'
-Run 
```python test.py```
