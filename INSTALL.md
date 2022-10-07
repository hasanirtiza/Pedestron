## Installation

### Requirements

- Linux
- Python 3.5+ ([Say goodbye to Python2](https://python3statement.org/))
- PyTorch 1.0+ or PyTorch-nightly (We have tested Pedestron using  PyTorch==1.4.0)
- CUDA 9.0+
- NCCL 2+
- GCC 4.9+
- [mmcv](https://github.com/open-mmlab/mmcv) (We have tested Pedestron using  mmcv==0.2.10 and 0.2.14)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC: 4.9/5.3/5.4/7.3



### Install mmdetection

a. Create a conda virtual environment and activate it. Then install Cython.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install cython
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/).

c. Clone the Pedestron repository.

```shell
git clone https://github.com/hasanirtiza/Pedestron.git
cd Pedestron
```

d. Install Pedestron (other dependencies will be installed automatically).

1. If your environment is cuda 10.2+/11, you need **modify Pedestron code to change all AT_CHECK to TORCH_CHECK**, then the following command can compile successfully.

2. If your environment is cuda 9.0/9.2/10.0, just run the following command.
   
```shell
python setup.py develop
# or "pip install -v -e ."
```

Note:

1. It is recommended that you run the step e each time you pull some updates from github. If there are some updates of the C/CUDA codes, you also need to run step d.
The git commit id will be written to the version number with step e, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, mmdetection is installed on `dev` mode, any modifications to the code will take effect without installing it again.



### Docker Installation

1) Setup environment by docker
 - Requirements: Install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
 - Create docker image:
 ```shell
sudo docker build . -t pedestron
```
 - Run docker image:
```shell
sudo docker run --gpus all --shm-size=8g -it --rm pedestron
```


### CUDA 11+ instllation

1) Please follow the tips and instructions given at this [PR request](https://github.com/hasanirtiza/Pedestron/pull/150). 

### Prepare COCO dataset.

It is recommended to symlink the dataset root to `$MMDETECTION/data`.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```

### Scripts
[Here](https://gist.github.com/hellock/bf23cd7348c727d69d48682cb6909047) is
a script for setting up mmdetection with conda.

### Notice
You can run `python(3) setup.py develop` or `pip install -v -e .` to install mmdetection if you want to make modifications to it frequently.

If there are more than one mmdetection on your machine, and you want to use them alternatively.
Please insert the following code to the main file
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```
or run the following command in the terminal of corresponding folder.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```
