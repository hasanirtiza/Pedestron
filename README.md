# **Updates**
We are currently updating this repository. We are preparing training and testing launchers, which will be pushed soon. 
We have added a pre-trained model for CityPersons. 


# Pedestron
[Pedestron](https://128.84.21.199/pdf/2003.08799.pdf) is an [MMetection](https://github.com/open-mmlab/mmdetection) based repository that focuses on the advancement of reserach on pedestrian detection.
We provide processed annotations and scripts to process the annotation of different pedestrian detection benchmarks.


### Installation
We refer to the installation and list of dependencies to [installation](https://github.com/hasanirtiza/Pedestron/blob/master/INSTALL.md) file.
Clone this repo and follow [installation](https://github.com/hasanirtiza/Pedestron/blob/master/INSTALL.md).

### List of detectors

*Currently we provide configurations for Cascade Mask-RCNN  


### Following datasets are currently supported 
* [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)
* [CityPersons](https://bitbucket.org/shanshanzhang/citypersons/src/default/)
* [EuroCity Persons](https://eurocity-dataset.tudelft.nl/)
* [CrowdHuman](https://www.crowdhuman.org/)
* [WiderPedestrian Challenge](https://competitions.codalab.org/competitions/20132)


### Preparation
1. Download the datasets from the official sites. Fill in the copyright forms where applicable. 
2. Place them in ./datasets folder in the follwoing heararchy, for example annotation file for CityPersons should be (./datsets/CityPersons/) and images should be
 (./datsets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/train) for training images and collapse all validtaion images into(./datsets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder/). 
  
3. Use our pre-processing script to convert the annotations into Pedestron acceptable format.


### Pre-Trained models
1) [CityPersons](https://drive.google.com/open?id=1B487ljaU9FxTSFaLoirOSqadZ-39QEH8)
2) Caltech [link]
3) EuroCity Persons [link]


### Testing demo for CityPersons
1) Download the pretrained [CityPersons](https://drive.google.com/open?id=1B487ljaU9FxTSFaLoirOSqadZ-39QEH8) model and place it in the folder "models_pretrained/".
2) Run the following command:

```shell 
python ./tools/test_city_person.py configs/elephant/cityperson/cascade_hrnet.py ./models_pretrained/epoch_ 5 6\
 --out result_citypersons.json --mean_teacher 
```

### Training
Coming Soon

### Testing
Coming Soon

### Please cite the following work
[ArXiv version](https://128.84.21.199/pdf/2003.08799.pdf)
```
@article{irtiza20elephant,
  title={Pedestrian Detection: The Elephant In The Room},
  author={Hasan, Irtiza and Liao, Shengcai and Li, Jinpeng and Akram, Saad Ullah and Ling, Shao},
  journal={arXiv preprint},
  year={2020}}
```
