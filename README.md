# KI-basierte Detektion und Sortierung von Abfallstoffen mit dem Robotergreifarm Niryo Ned2
##  (AI-based detection and sorting of trash with the robot Niryo Ned2)
![output](https://github.com/2bits2/recyclable/assets/76791368/2e2ffca6-ed9f-4def-bc4b-eac034019bd7)

  
### Main Idea
The main problem with training a trash segmentation model is 
that a good model needs massive high quality data of trash to be trained on.
There exist different publically available datasets for trash but in different
formats.
For example for trash classification there are the datasets like TrashNet and TrashBox.
For object detection of trash you might want to use garbageclassificationv3.
Common Trash segmentation datasets are TACO and Zerowaste.

With new Foundation Models like GroundingDino and SAM it might be possible
to convert and combine many different already existing datasets into a one unified dataset of trash 
without annotating everything by hand again.

A yolov8 segmentaion model trained on this newly generated dataset might be   
feasable to sort trash with the Niryo Ned2 robot.

<img src="https://github.com/2bits2/recyclable/assets/76791368/e6fbadee-b15a-4e6f-8c27-2c2a1d96cd13.jpg" width="500px" />

As the detection of Trash with GroundingDino is limited, 
only the trash object with the highest confidence score is taken for segmentation.
The image is then trimmed to only contain that object.

To adapt the newly generated dataset more to the actual environment it might 
useful to take multiple objects randomly and place them randomly on different backgrounds of the Niryo Ned2 conveyor belt.

<img src="https://github.com/2bits2/recyclable/assets/76791368/26e04d35-5b9e-4ee6-83b0-d5f6eeab468e.jpg" width="700px" />


### Segmentation and Sorting of Trash
robot.py contains functionality to
control a niryo ned 2 robot to sort objects (trash in this case)

### Simple 2-Stage Framework vs 1-Stage Framework
segment.py is there to combine a really simple hsv color thresholding method (or yolov8 segmentation models) 
with yolov8 classification models
to compare different configuration / training methods. 

### Dataset conversion
dataset.py contains functionality to:

- to automatically convert a classification dataset (like TrashBox) with GroundingDino and SAM into a yolov8 segmentation dataset
  (as there only exist a few segmentation datasets of trash)
  ```python
  cls2seg(grounding_dino_model,
            sam_predictor,
            src_dir,
            dst_dir,
            category_to_label_map,
            box_threshold = 0.35,
            text_threshold = 0.25)
  ```
  
- to automatically convert a yolov8 object detection dataset into a yolov8 segmentation dataset with SAM 
  ```python
    objectdet2seg(
      sam_predictor,
      object_detection_directory,
      output_segmentation_directory,
      box_threshold = 0.35,
      text_threshold = 0.25,
      standard_size=[224, 224])
  ```
  
- to convert a segmentation dataset with one object per image into a dataset
  that resembles the Niryo Ned2 surrounding more closely
  (using an image implanting technique with Niryo Ned2 conveyor belt background images)
  ```python
     seg_transform(src_dir, dst_dir, get_background, max_object_count, image_prefix="", seed=3)
  ```
  
- to combine multiple segmentation datasets with different labels per category into a unified dataset
  where the labels are renamed properly per category
  (and optionally augmented with images that are randomly transplanted into background images) 
  ```python
    combine_segs(datasetname2names, output_path, name2intcategory, weights,
                      get_background=None, max_object_count=3, seed=3)
    ```
### Evaluation
to evaluate the performance of trained and combined models on test data, the evaluate.py file 
is there to generate confusion matrices and mean average precision







