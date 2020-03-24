# **Notes**
* We provide unoffical processd annotations for Caltech and CityPersons datasets.
* For images and original annotations of these two datasets along with other datasets, you need to visit the official pages of the respective datasets, fill sign up forms where applicable.  
* Cite the respective sources for each datasets

 ### Preparation

- **CityPersons**
    1. We provide unofficial annotations in .json format for CityPerson already contained in the repository ./datasets/CityPersons
	2. Download the datasets from the official sites. Fill in the copyright forms where applicable. For CityPersons official annotaions visit the following [link](https://bitbucket.org/shanshanzhang/citypersons/src/default/) and for images visit this [link](https://www.cityscapes-dataset.com/)     	
	3. Place CityPersons in ./datasets folder in the follwoing heararchy, for example annotation file for CityPersons should be (./datsets/CityPersons/) and images should be
   (./datsets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/train) for training images and collapse all validtaion images into(./datsets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder/). 

- **Caltech**
   1. We provide unofficial annotations in .json format for Caltech already contained in the repository ./datasets/Caltech
   2. Download the datasets from the official sites. Fill in the copyright forms where applicable. For Caltech **new annotaions** visit the following [link](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/people-detection-pose-estimation-and-tracking/how-far-are-we-from-solving-pedestrian-detection/) and for images visit this [link](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)     	
   3. Place Caltech in ./datasets folder in the follwoing heararchy, for example annotation file for Caltech should be (./datsets/Caltech/) and extract images and images should be
   (./datsets/Caltech/train_images) for training images and all validtaion images into(./datsets/Caltech/test_images). 
   4. For evaluation on Caltech tools/caltech/convert_json_to_txt.py is used to convert the caltech results from json to .txt which can be evaluated using the official [caltech matlab evalutation tool](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/). 

- **EuroCity Persons**
   1. Download the datasets from the official sites. Fill in the copyright forms where applicable. Images and annotaions (day-time) can be downloaded from this [link](https://eurocity-dataset.tudelft.nl/)
   2. Convert the annotations to Pedestron format from the provided scripts in tools/convert_datasets/ecp/
   3. Place EuroCity Person in ./datasets folder in the follwoing heararchy, for example annotation file for EuroCity Persons should be (./datasets/EuroCity/) and images should be
   (./datsets/EuroCity/ECP/day/img/train) for training images and all validtaion images into(./datsets/EuroCity/ECP/day/img/val).   

- **Custom Dataset**
	1. Pedestron accepts annotations in the .json format similar to COCO annotation style.
	2. See one of the annotations file to see what are the required fields and bounding box styles accepted by Pedestron. 
	
	
### Tools to pre-process annotations

* Some sample conversion scripts to Pedestron style can be seen at tools/convert_datasets/
