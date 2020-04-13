
# Pedestron
<img title="Amsterdam" src="gifs/1.gif" width="400" /> <img title="Amsterdam" src="gifs/2.gif" width="400"/>
<img title="Amsterdam" src="gifs/3.gif" width="400"/> <img title="Amsterdam" src="gifs/4.gif" width="400"/>


[Pedestron](https://128.84.21.199/pdf/2003.08799.pdf) is a [MMetection](https://github.com/open-mmlab/mmdetection) based repository that focuses on the advancement of research on pedestrian detection.
We provide processed annotations and scripts to process the annotation of different pedestrian detection benchmarks.

# **Updates**
* [NEW] **Added configurations and pre-trained model for Hybrid Task Cascade (HTC)**
* [NEW] **Added backbone MobileNet along with its benchmarking**
* [NEW] **Evaluation code for the Caltech dataset, added to the repository**


### YouTube
* YouTube [link](https://www.youtube.com/watch?v=cemN7JbgxWE&feature=youtu.be)  for qualitative results on Caltech. Pre-Trained model available.

### Installation
We refer to the installation and list of dependencies to [installation](https://github.com/hasanirtiza/Pedestron/blob/master/INSTALL.md) file.
Clone this repo and follow [installation](https://github.com/hasanirtiza/Pedestron/blob/master/INSTALL.md).

### List of detectors
Currently we provide configurations for with different backbones
* Cascade Mask-R-CNN
* Faster R-CNN
* RetinaNet
* Hybrid Task Cascade (HTC)  


### Following datasets are currently supported 
* [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)
* [CityPersons](https://bitbucket.org/shanshanzhang/citypersons/src/default/)
* [EuroCity Persons](https://eurocity-dataset.tudelft.nl/)
* [CrowdHuman](https://www.crowdhuman.org/)
* [WiderPedestrian Challenge](https://competitions.codalab.org/competitions/20132)


### Datasets Preparation
* We refer to [Datasets preparation file](Datasets-PreProcessing.md) for detailed instructions




### Benchmarking of Pre-Trained models
|    Detector                | Dataset   | Backbone| Reasonable  | Heavy    | 
|--------------------|:--------:|:--------:|:--------:|:--------:|
| [Cascade Mask R-CNN](https://drive.google.com/open?id=1B487ljaU9FxTSFaLoirOSqadZ-39QEH8) | CityPersons        | HRNet | 7.5        |   28.0      | 
| [Cascade Mask R-CNN](https://drive.google.com/open?id=1ysWlzN92EInIjDD_QwIq3iQEZJDo8wtT) | CityPersons        | MobileNet | 10.2        |   37.3      | 
| [Faster R-CNN](https://drive.google.com/open?id=1aanqAEFBc_KGU8oCFCji-wqmLmqTd749) | CityPersons        | HRNet | 10.2        |   36.2      |
| [RetinaNet](https://drive.google.com/open?id=1MGxZitqLzQtd2EF8cVGYNzSKt73s9RYY) | CityPersons        | ResNeXt | 14.6        |   39.5      |
| [Hybrid Task Cascade (HTC)](https://drive.google.com/open?id=1qPEJ1r48Ggl2TdE1ohcDoprZomC2j3SX) | CityPersons        | ResNeXt | 9.5       |   35.8      | 
| [Cascade Mask R-CNN](https://drive.google.com/open?id=1HkoUPlONSF04AKsPkde4gmDLMBf_vrnv) | Caltech        | HRNet |   1.7      |    25.7     | 
| [Cascade Mask R-CNN](https://drive.google.com/open?id=1GzB3O1JxPN5EusJSyl7rl9h0sQAVKf15) | EuroCity Persons | HRNet |    4.4     |  21.3       | 
 

### Pre-Trained models
Cascade Mask R-CNN
1) [CityPersons](https://drive.google.com/open?id=1B487ljaU9FxTSFaLoirOSqadZ-39QEH8)
2) [Caltech](https://drive.google.com/open?id=1HkoUPlONSF04AKsPkde4gmDLMBf_vrnv)
3) [EuroCity Persons](https://drive.google.com/open?id=1GzB3O1JxPN5EusJSyl7rl9h0sQAVKf15)

Faster R-CNN
1) [CityPersons](https://drive.google.com/open?id=1aanqAEFBc_KGU8oCFCji-wqmLmqTd749)

RetinaNet
1) [CityPersons](https://drive.google.com/open?id=1MGxZitqLzQtd2EF8cVGYNzSKt73s9RYY)

Hybrid Task Cascade (HTC)
1) [CityPersons](https://drive.google.com/open?id=1qPEJ1r48Ggl2TdE1ohcDoprZomC2j3SX)

### Running a demo using pre-trained model on few images
1) Pre-trained model can be evaluated on sample images in the following way

```shell 
python tools/demo.py config checkpoint input_dir output_dir
```
Download one of our provided pre-trained model and place it in  models_pretrained folder. Demo can be run using the following command

```shell 
python tools/demo.py configs/elephant/cityperson/cascade_hrnet.py ./models_pretrained/epoch_5.pth.stu demo/ result_demo/ 
```



### Training
Train with single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

Train with multiple GPUs
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

For instance training on CityPersons using single GPU 

```shell
python tools/train.py configs/elephant/cityperson/cascade_hrnet.py
```

Training on CityPersons using multiple(7 in this case) GPUs 
```shell
./tools/dist_train.sh configs/elephant/cityperson/cascade_hrnet.py 7  
```

### Testing

Test can be run using the following command.
 
```shell 
python ./tools/TEST_SCRIPT_TO_RUN.py PATH_TO_CONFIG_FILE ./models_pretrained/epoch_ start end\
 --out Output_filename --mean_teacher 
``` 

For example for CityPersons inference can be done the following way

1) Download the pretrained [CityPersons](https://drive.google.com/open?id=1B487ljaU9FxTSFaLoirOSqadZ-39QEH8) model and place it in the folder "models_pretrained/".
2) Run the following command:

```shell 
python ./tools/test_city_person.py configs/elephant/cityperson/cascade_hrnet.py ./models_pretrained/epoch_ 5 6\
 --out result_citypersons.json --mean_teacher 
```
* Similarly change respective paths for EuroCity Persons
* For Caltech refer to [Datasets preparation file](Datasets-PreProcessing.md)

### Please cite the following work
[ArXiv version](https://128.84.21.199/pdf/2003.08799.pdf)
```
@article{hasan2020pedestrian,
  title={Pedestrian Detection: The Elephant In The Room},
  author={Hasan, Irtiza and Liao, Shengcai and Li, Jinpeng and Akram, Saad Ullah and Shao, Ling},
  journal={arXiv preprint arXiv:2003.08799},
  year={2020}
}
```
