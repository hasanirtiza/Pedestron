# **Notes**
* We provide unoffical processd annotations (.json format) for Caltech, CityPersons, EuroCity Persons, WiderPedestrian Challenge and CrowdHuman datasets.
* For images and original annotations of these two datasets along with other datasets, you need to visit the official pages of the respective datasets, fill sign up forms where applicable.  
* Cite the respective sources for each datasets.

 ### Preparation

- **CityPersons**
    1. We provide unofficial annotations in .json format for CityPerson already contained in the repository `./datasets/CityPersons`.
	2. Download the datasets from the official sites. Fill in the copyright forms where applicable. For CityPersons official annotaions visit the following [link](https://bitbucket.org/shanshanzhang/citypersons/src/default/) and for images visit this [link](https://www.cityscapes-dataset.com/).	
	3. Place CityPersons in `./datasets` folder in the following heararchy, for example annotation file for CityPersons should be (`./datsets/CityPersons/`) and images should be
   (`./datsets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/train`) for training images and collapse all validtaion images into(`./datsets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder/`). 

- **Caltech**
   1. We provide unofficial annotations in .json format for Caltech already contained in the repository `./datasets/Caltech`.
   2. Download the datasets from the official sites. Fill in the copyright forms where applicable. For Caltech **new annotaions** visit the following [link](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/people-detection-pose-estimation-and-tracking/how-far-are-we-from-solving-pedestrian-detection/) and for images visit this [link](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/).     	
   3. Place Caltech in ./datasets folder in the follwoing heararchy, for example annotation file for Caltech should be (`./datsets/Caltech/`) and extract images and images should be
   (`./datsets/Caltech/train_images`) for training images and all validtaion images into(./datsets/Caltech/test_images). 
   4. For evaluation on Caltech, start by converting .json output of Pedestron to .txt using `tools/caltech/convert_json_to_txt.py`. The output will be saved in `tools/caltech/eval_caltech/Pedestron_Results` with the following hierarchy `Pedestron_Results/epoch_number/set_number/*.txt`. 
   5. Open Matlab (evaluation code is only available in Matlab at the momment and borrowed from https://github.com/liuwei16/CSP) and navigate Matlab to the folder tools/caltech/eval_caltech. Run `dbEval.m`, results will be saved at `eval_caltech/ResultsEval/eval-newReasonable.txt`.       

- **EuroCity Persons**
   1. We provide unofficial annotations in .json format for EuroCity Persons already contained in the repository `./datasets/EuroCity`.
   2. Download the datasets from the official sites. Fill in the copyright forms where applicable. Images and annotaions (day-time) can be downloaded from this [link](https://eurocity-dataset.tudelft.nl/).
   3. Place EuroCity Person in ./datasets folder in the follwoing heararchy, for example annotation file for EuroCity Persons should be (`./datasets/EuroCity/`) and images should be
   (`./datsets/EuroCity/ECP/day/img/train`) for training images and all validtaion images into(`./datsets/EuroCity/ECP/day/img/val`). 

- **WiderPedestrian Challenge**
    1. We provide unofficial annotations in .json format for WiderPedestrian Challenge already contained in the repository `./datasets/Wider_challenge`
	2. Download the datasets from the official sites. Fill in the copyright forms where applicable. For WiderPedestrian Challenge official annotaions and images visit the following [link](https://competitions.codalab.org/competitions/20132).     	
	3. Place WiderPedestrian Challenge in `./datasets` folder in the following heararchy, for example annotation file for WiderPedestrian Challenge should be (`./datsets/Wider_challenge/`) and images should be
   (`./datsets/Wider_challenge/train_images`) for training images.

- **CrowdHuman**
    1. We provide unofficial annotations in .json format for CrowdHuman already contained in the repository `./datasets/CrowdHuman`
	2. Download the datasets from the official sites. Fill in the copyright forms where applicable. For CrowdHuman official annotaions and images visit the following [link](https://www.crowdhuman.org/).     	
	3. Place CrowdHuman in ./datasets folder in the following heararchy, for example annotation file for CrowdHuman should be (`./datsets/CrowdHuman/`) and images should be
   (`./datsets/CrowdHuman/Images`) for training images and (`./datsets/CrowdHuman/Images_val`) for validation images.    

- **Custom Dataset**
	1. Pedestron accepts annotations in the .json format similar to COCO annotation style.
	2. See one of the annotations file to see what are the required fields and bounding box styles accepted by Pedestron. 
	
	
### Tools to pre-process annotations

* Some sample conversion scripts to Pedestron style can be seen at `tools/convert_datasets/`.
