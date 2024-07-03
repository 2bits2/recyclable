# KI-basierte Detektion und Sortierung von Abfallstoffen mit dem Robotergreifarm Niryo Ned2
##  (AI-based detection and sorting of trash with the robot Niryo Ned2)

This repository contains functionality to:
- control a niryo ned 2 robot to sort objects (trash in this case) (robot.py, resources/poses.json)
  ![output](https://github.com/2bits2/recyclable/assets/76791368/2e2ffca6-ed9f-4def-bc4b-eac034019bd7)
- to combine yolov8 classification with segmentation models
  (to compare a 1-stage segmentation method with a simple 2-stage segmentation method)
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
    def combine_segs(datasetname2names, output_path, name2intcategory, weights,
                      get_background=None, max_object_count=3, seed=3)
    ```
