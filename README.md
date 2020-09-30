# Convert-YOLO-to-PascalVOC

Last Page Update: **20/07/2020**

A python script to convert YOLO into Pascal VOC 2012 format. It generates xml annotation file in PASCAL VOC format for Object Detection.

<p align="center"><img src="https://raw.githubusercontent.com/carolinepacheco/Convert-COCO-to-PascalVOC/master/docs/convert.png" border="0" /></p>


## Notes
 
 * Make sure you have the dependencies listed on yolo_to_voc.py. 
 
 * Update root path (where this script lies) in line 46. ``ROOT = 'coco'``. 
 
 * Let's say that you have a custom dataset, which is not included in COCO. eg ship. Add its name to ``YOLO_CLASSES=()``, in the first position.
 
 * Remove images that are already in /coco/images, /coco/labels and /coco/outputs.
 
 * Put all your images at /coco/images folder.
 
 * Put corresponding annotations (.txt files) to /coco/labels
 
 
##  Prerequisites (my environment)

* Python 3.8.3

* Numpy

* Opencv 

 
 ## Usage
 
 Please to run this script use the command below :
 
```
python3 yolo_to_voc.py
```
 
 or 
 
```
python yolo_to_voc.py

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

