# Convert-YOLO-to-PascalVOC

Last Page Update: **21/12/2023**

A python script to convert YOLO into Pascal VOC 2012 format. It generates xml annotation file in PASCAL VOC format for Object Detection.

<p align="center"><img src="https://raw.githubusercontent.com/carolinepacheco/Convert-COCO-to-PascalVOC/master/docs/convert.png" border="0" /></p>


## Notes
 
 * Make sure you have the dependencies listed on yolo_to_voc.py. 

 * The scripts works within a base folder context where the yolo dataset is. by default, the base folder is "yolo".

 * By default, is assumed that the images are at yolo/images

 * By default, is assumed that the labels are at yolo/labels

 * By default, script stores outputs in yolo/pascal
 
##  Tested environment

* python `>=3.6.x`

* numpy `==1.19.5`

* opencv-python `==4.6.0.66`

* lxml `==4.9.2`

 
## Usage
```shell
$python3 yolo_to_voc.py --help
usage: yolo_to_voc.py [-h] [--base_folder BASE_FOLDER]
                      [--output_folder OUTPUT_FOLDER]
                      [--label_folder LABEL_FOLDER]
                      [--image_folder IMAGE_FOLDER] [--class_map CLASS_MAP]

Convert YOLO annotations to PASCAL VOC format

optional arguments:
  -h, --help            show this help message and exit
  --base_folder BASE_FOLDER
                        Folder to search for YOLO dataset. defaults to "yolo"
  --output_folder OUTPUT_FOLDER
                        Output folder name to store the PASCAL annotations. defaults to {base_folder}/pascal
  --label_folder LABEL_FOLDER
                        Folder containing labels in YOLO format. defaults to {base_folder}/labels
  --image_folder IMAGE_FOLDER
                        Folder containing dataset images. defaults to {base_folder}/images
  --class_map CLASS_MAP
                        Path to json file where an object of type { number: string } stores class values for custom dataset. defaults of all coco classes
```

#### Example class map JSON file:
```
{
  "10": "person",
  "11": "bicycle",
  "20": "car",
  "30": "motorcycle",
}
```

Some academic projects
-------------------------
```
https://www.behance.net/carolinepacheco
```

Medium blog site
-------------------------
```
https://medium.com/@lolyne.pacheco
```
