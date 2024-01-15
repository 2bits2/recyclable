import dataset
#from torch import BoolTensor, IntTensor, Tensor
#from torchmetrics.detection.mean_ap import MeanAveragePrecision
import json
import yaml
import os
import glob
import cv2
import numpy as np
from functools import partial
#import matplotlib as plt
from ultralytics import YOLO
#import torch

import ultralytics
import supervision as sv
import time
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

# a little hack to serialize tensors and numpy
# arrays , that would otherwise
# not be serialized
class NumpyAndTensorEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, tuple):
                return list(obj)
        elif isinstance(obj, Tensor):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# uses torchmetrics
# to evaluate a predictions
def compute_evaluation(image_filenames,
                annotation_filenames,
                isbox,
                names,
                prediction_func,
                class_metrics,
                extended_summary,
                map_to_object_class=False):

    # there is an annotation file
    # for each image
    if len(image_filenames) != len(annotation_filenames):
        print("each image file shall have a corresponding annotation file")
        return None

    # if the dataset is only
    # for object detection but
    # not for segmentation
    # we can just compute the
    # bounding box metric
    # otherwise we can also compute
    # additional segmentation metrics
    if isbox:
        metric_type = "bbox"
        print("choosing box metric")
    else:
        metric_type = ("bbox","segm")
        print("choosing bbox and segm metric")

    # for the calculation we use
    # the torchmetrics functionality
    metric = MeanAveragePrecision(
        iou_type=metric_type,
        class_metrics=class_metrics,
        extended_summary=extended_summary
    )

    # save how long it took
    # for each prediction
    total_times = []

    for image_path, annotation_path in zip(image_filenames, annotation_filenames):

        # load the image
        image = cv2.imread(image_path)
        image_dimension = image.shape[:2]

        # get the prediction
        # and ground truth values
        prediction = prediction_func(image)
        ground_truth = dataset.load_segmentation_ground_truth(
            image_dimension,
            annotation_path,
            isbox
        )

        # we can map everything to just one class
        # if we just want find anything
        # and don't care what object it is
        if map_to_object_class:
            prediction["labels"] = [0 for i in range(0, len(prediction["labels"]))]
            ground_truth["labels"] = [0 for i in range(0, len(ground_truth["labels"]))]

        # store the time
        # seperately for later
        # evaluation
        total_times.append(prediction["time_total"])

        ## visualize
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # for i, m  in enumerate(prediction["masks"]):
        #     print(prediction["labels"][i])
        #     cv2.imshow("prediction", m)
        #     cv2.waitKey(0)
        # for i, m in enumerate(ground_truth["masks"]):
        #     print(ground_truth["labels"][i])
        #     cv2.imshow("ground truth", m)
        #     cv2.waitKey(0)


        # torchmetrics expects
        # the input in Tensors
        prediction = {
            "boxes": Tensor(np.array(prediction["boxes"])),
            "labels": IntTensor(np.array(prediction["labels"])),
            "masks": BoolTensor(np.array(prediction["masks"])),
            "scores": Tensor(np.array(prediction["scores"]))
        }

        ground_truth = {
            "boxes": Tensor(np.array(ground_truth["boxes"])),
            "labels": IntTensor(np.array(ground_truth["labels"])),
            "masks": BoolTensor(np.array(ground_truth["masks"]))
        }

        # now we can pass all the data
        # to torchmetrics
        metric.update([prediction], [ground_truth])

    # now we can compute everything
    result = metric.compute()

    fig, ax = metric.plot()
    #ax.set_fontsize(fs=20)
    #fig.set_title("This is a nice plot")
    fig.savefig("my_awesome_plot.png")

    result["time_total"] = total_times
    return result




def test_segmentation(segment_func, dataset_yaml_file, safe_file, map_to_object_class=False):
    with open(dataset_yaml_file, "r") as f:
        try:
            dataset_info = yaml.safe_load(f)
        except yaml.YAMLError as err:
            print(err)
            return

    full_path = os.path.abspath(dataset_yaml_file)
    dirname = os.path.dirname(full_path)

    images_subdir = dataset_info["test"]
    image_wildcard = f"{dirname}/{images_subdir}/*.jpg"

    # get the corresponding image
    # and annotation files
    image_names = glob.glob(image_wildcard)
    label_names = [
        image_name.replace(
            "/test/images/",
            "/test/labels/"
        ).replace(
            ".jpg",
            ".txt"
        ) for image_name in image_names
    ]

    result = compute_evaluation(image_names,
                         label_names,
                         isbox=False,
                         names=dataset_info["names"],
                         prediction_func=segment_func,
                         class_metrics=True,
                         extended_summary=False,
                         map_to_object_class=map_to_object_class)

    with open(safe_file, "w") as f:
        json.dump(result, f, cls=NumpyAndTensorEncoder)

    print(f"saved segmentation evaluation at {safe_file}")


def unzip_list(l):
    return list(map(list, zip(*l)))

import segment
import utils
import cv2

#from torchvision import ops

def cutoff_detection(cutoff, real_detect, image):
    img = image[cutoff:-cutoff, cutoff:-cutoff]
    detections = real_detect(img)
    if len(detections) == 0:
        return detections
    detections.xyxy[:,:] += cutoff
    masks = []
    for m in detections.mask:
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[cutoff:-cutoff, cutoff:-cutoff] = m[:,:]
        masks.append(mask)
    detections.mask = np.array(masks)
    return detections

def map_to_zero_detection(detections:sv.Detections):
    if len(detections) == 0:
        return detections
    detections.class_id.fill(0)
    return detections


def color_thresh_detect(image, color_hsv_min_max=[[0, 70, 0], [179, 255, 255]]) -> sv.Detections:
    start = time.time()
    mask = segment.color_threshold_mask(image, color_hsv_min_max)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    speed = time.time() - start


    if len(contours) == 0:
        detections = sv.Detections.empty()
        detections.speed = speed
        return detections

    object_boxes_xyxy = []
    object_masks = []
    confidences = []
    class_ids = []

    for i in range(len(contours)):
        object_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(object_mask, contours, i, 255, -1)
        #object_mask = object_mask.astype(bool)
        x1, y1, w, h = cv2.boundingRect(contours[i])
        x2 = x1 + w
        y2 = y1 + h
        object_boxes_xyxy.append([x1, y1, x2, y2])
        object_masks.append(object_mask)
        confidences.append(1)
        class_ids.append(0)

    xyxys = np.array(object_boxes_xyxy)
    #print(xyxys)

    detections = sv.Detections(
        xyxy=xyxys,
        mask=np.array(object_masks).astype(bool),
        confidence=np.array(confidences),
        class_id=np.array(class_ids)
    )

    detections.speed = speed
    return detections


def yolov8_classify(model, image) -> sv.Classifications:
    print(image.shape)
    return sv.Classifications.from_ultralytics(model(image)[0])

def combined_seg(segment_func, classify, image, class_mapping=None):
    start = time.time()
    detections: sv.Detections = segment_func(image)
    detconfidences = []
    for i in range(len(detections)):
        cropped_image = sv.crop_image(image, detections.xyxy[i])
        classifications : sv.Classifications = classify(cropped_image)
        class_ids, confidences = classifications.get_top_k(1)
        detections.class_id[i] = class_ids[0]
        detections.confidence[i] = confidences[0]

    if class_mapping is not None:
        if len(detections) > 0:
            detections.class_id = np.vectorize(class_mapping.get)(detections.class_id)

    speed = time.time() - start
    detections.speed = speed
    return detections

def seg2clsdataset(dataset:sv.DetectionDataset, output_dir):
    output_dir = os.path.abspath(output_dir)
    for name in dataset.classes:
        os.makedirs(f"{output_dir}/{name}", exist_ok=True)

    for i, (image_name, image) in enumerate(dataset.images.items()):
            annotated_image = image.copy()
            ground_truth_detections : sv.Detections = dataset.annotations[image_name]
            image_basename = os.path.basename(image_name)
            for d in range(len(ground_truth_detections)):
                cropped_image = sv.crop_image(image, ground_truth_detections.xyxy[d])
                name = dataset.classes[ground_truth_detections.class_id[d]]
                output_file = f"{output_dir}/{name}/{name}_{i}_{d}.jpg"
                cv2.imwrite(output_file, cropped_image)

def otsu_object_detect(image: np.ndarray) -> sv.Detections:
    start = time.time()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_threshold, binary_image = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda c: cv2.contourArea(c) > 200, contours))
    speed = time.time() - start

    if len(contours) == 0:
        detections = sv.Detections.empty()
        detections.speed = speed
        return detections

    class_ids = np.repeat(0, len(contours))
    masks = []
    xyxys = []
    for i in range(len(contours)):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, i, 255, -1)
        x, y, w, h = cv2.boundingRect(contours[i])
        xyxy = np.array([x, y, x+w, y+h])
        xyxys.append(xyxy)
        masks.append(mask.astype(bool))
    xyxys = np.array(xyxys)
    masks = np.array(masks)
    conficence = np.repeat(0.9, len(class_ids))

    detections = sv.Detections(xyxys, masks, class_id=class_ids, confidence=conficence)
    detections.speed = speed
    return detections


def yolov8_detect(model:YOLO, image:np.ndarray, class_mapping=None):

    start = time.time()
    results : ultralytics.engine.results.Results = model(image)[0]
    speed = time.time() - start
    detections =  sv.Detections.from_ultralytics(results)
    detections.speed = speed * 1000

    if len(detections) == 0:
        return detections

    if class_mapping is not None:
        detections.class_id = np.vectorize(class_mapping.get)(detections.class_id)

    return detections



def get_class_mapping(modelnamedict, datasetclassnames):
    class_mapping = modelnamedict.copy()
    for cls in class_mapping:
        class_mapping[cls] = datasetclassnames.index(class_mapping[cls])
    return class_mapping

def get_single_class_mapping(modelnamedict):
    class_mapping = modelnamedict.copy()
    for cls in class_mapping:
        class_mapping[cls] = 0
    return class_mapping

#import matplotlib

def evaldataset(target_dirname, dataset, detection_callback,
                gen_confusion_matrix=True,
                gen_map=True,
                gen_annotations=False):

    target_dirname = os.path.abspath(target_dirname)
    target_image_dirname = os.path.join(target_dirname, "images")
    os.makedirs(target_image_dirname, exist_ok=True)

    if gen_confusion_matrix:
        confusion_matrix = sv.ConfusionMatrix.benchmark(
            dataset = dataset,
            callback = detection_callback)

        confusion_matrix.plot(f"{target_dirname}/confusion_matrix.jpg")
        fig = confusion_matrix.plot(f"{target_dirname}/confusion_matrix_normalized.jpg", normalize=True)
        with open(f"{target_dirname}/confusion_matrix.json", "w") as f:
            json.dump({"iou_threshold": confusion_matrix.iou_threshold,
                       "conf_threshold": confusion_matrix.conf_threshold,
                       "classes": confusion_matrix.classes,
                       "conf_matrix": confusion_matrix.matrix
                       }, f, cls=NumpyAndTensorEncoder)

    if gen_map:
        mean_average_precision = sv.MeanAveragePrecision.benchmark(
            dataset=dataset,
            callback=detection_callback)

        map_result = {
            "map50_95": mean_average_precision.map50_95,
            "map75": mean_average_precision.map75,
            "map50": mean_average_precision.map50,
            "per_class_ap_50_95": mean_average_precision.per_class_ap50_95
        }
        with open(f"{target_dirname}/map_result.json", "w") as f:
            json.dump(map_result, f, cls=NumpyAndTensorEncoder)


    if gen_annotations:
        gt_color = sv.Color(10, 225, 10)
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_LEFT, text_scale=0.2)
        gt_label_annotator = sv.LabelAnnotator(color=gt_color,
                                               text_position=sv.Position.BOTTOM_CENTER,
                                               text_scale=0.2)

        detected_mask_annotator = sv.MaskAnnotator(
            opacity=0.5,
            color_lookup=sv.ColorLookup.INDEX)
        ground_truth_mask_annotator =  sv.PolygonAnnotator(color=gt_color, thickness=2)

        speeds = []
        basenames = []
        annotated_images = []
        empty_non_background_images = []
        for i, (image_name, image) in enumerate(dataset.images.items()):

            annotated_image = image.copy()
            inferred_detections = detection_callback(image)
            ground_truth_detection = dataset.annotations[image_name]
            annotated_image = detected_mask_annotator.annotate(annotated_image,
                                                        inferred_detections)
            annotated_image = ground_truth_mask_annotator.annotate(annotated_image,
                                                  ground_truth_detection)
            annotated_image = label_annotator.annotate(annotated_image, inferred_detections)
            annotated_image = gt_label_annotator.annotate(annotated_image, ground_truth_detection)

            image_basename = os.path.basename(image_name)

            annotated_images.append(annotated_image)
            speeds.append(inferred_detections.speed)
            basenames.append(image_basename)

            if len(ground_truth_detection) > 0 and len(inferred_detections) == 0:
                empty_non_background_images.append(image_basename)

        with open(os.path.join(target_dirname, "speed_result.json"), "w") as f:
            json.dump({
                "names": basenames,
                "speeds": speeds,
            }, f)

        with open(os.path.join(target_dirname, "emptynonbackgroundimages.json"), "w") as f:
            json.dump({
                "names": empty_non_background_images
            }, f)

        for image_basename, annotated_image in zip(basenames, annotated_images):
            if not cv2.imwrite(os.path.join(target_image_dirname, image_basename), annotated_image):
                print(f"couldnt write {os.path.join(target_image_dirname, image_name)}")


# https://www.kaggle.com/code/gauthamupadhyaya/classification-using-yolov8
import random
import math
def get_rgb_image(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image
def test_accuracy_yolo(model, test_dir, plot_random=False, n_samples = 8):

    test_dir = os.path.abspath(test_dir)
    test_folders = os.listdir(test_dir)

    i = 0
    v = 0
    result_dict = {}


    matrix = {}
    for truth in test_folders:
        matrix[truth] = {}
        for pred in test_folders:
            matrix[truth][pred] = 0

    for folder in test_folders:
        results = model(f'{test_dir}/{folder}', verbose=False)
        result_dict.update({folder:results})
        for result in results:
            i += 1
            top1 = result.probs.top1
            classes = result.names
            top1_class_name = classes[top1]

            if top1_class_name == folder:
                v+=1

            matrix[folder][top1_class_name] += 1

    conf_matrix = np.zeros((len(test_folders), len(test_folders)))
    for a, t in enumerate(matrix):
        for b, p in enumerate(matrix[t]):
            conf_matrix[a][b] = matrix[t][p]

    if plot_random == True:
        c = 4
        r = math.ceil(n_samples/c)
        plt.figure(figsize=(20,5*r+1))
        #plt.suptitle(f'Visualizing random values with their labels.\nThe accuracy on the data is passed {str("%.2f" % (test_accuracy*100))}%')
        for i in range(n_samples):
            random_label = random.choice(list(result_dict.keys()))
            random_path = f'{test_dir}/{random_label}'
            pred_vals = result_dict[random_label]
            random_result = random.choice(pred_vals)
            random_top1 = random_result.probs.top1
            classes = random_result.names
            random_top1_class_name = classes[random_top1]
            random_top1cont = random_result.probs.top1conf.tolist()

            plt.subplot(r,c,i+1)
            plt.imshow(get_rgb_image(random_result.orig_img))
            plt.ylabel(f'Actual : {random_label}')
            plt.xlabel(f'Predicted : {random_top1_class_name}')
            plt.title(f'Confidence Interval : {str("%.2f" % random_top1cont)}')

    return (conf_matrix, test_folders)


def confmatrix_plot(output_pdf, conf_matrix, names):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    conf_df = pd.DataFrame(conf_matrix, index=names, columns=names)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6, 6))

    colors = sns.color_palette("colorblind")
    sns.heatmap(conf_df, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.set_size_inches(7,5)
    fig.set_dpi(100)
    plt.yticks(rotation=0)

    conf_matrix_normalized = conf_matrix / conf_matrix.astype(np.float32).sum(axis=0)

    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f',
                cmap='Blues', xticklabels=names,
                yticklabels=names, ax=ax2)
    ax2.set_xlabel("Predicted Label")
    #ax2.set_ylabel("True Label")
    ax2.set_title("Normalized")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    fig.set_size_inches(7,5)
    fig.set_dpi(100)
    plt.yticks(rotation=0)
    plt.show()
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight')



def dataset_numofinstances(dataset):
    classes = dataset.classes.copy()
    numofinstances = np.zeros(len(classes), dtype=np.uint)
    for image_name, image, annotation in dataset:
        numofinstances[annotation.class_id] += 1
    return (classes, numofinstances)


def numofinstance_barplot(output_pdf, classes, numofinstances, title, figsize=(8,6)):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    colors = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        x=classes,
        hue=classes,
        y=numofinstances,
        palette=colors,
        legend=False,
        ax=ax)

    for i, v in enumerate(numofinstances):
        ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=12, color='black')

    ax.set_xlabel("Kategorien", labelpad=10)
    ax.set_ylabel("Instanzen", labelpad=10)
    ax.set_title(title, pad=20)
    ax.tick_params(axis='both', which='both', length=0)



    maxnumofinstances = max(numofinstances)
    ax.set_ylim(top=maxnumofinstances + maxnumofinstances * 0.2 )
    fig.set_size_inches(7,5)
    fig.set_dpi(100)
    plt.show()
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight')


def torch_evaluate_dataset(dataset:sv.DetectionDataset, prediction_callback, metric_type=("bbox", "segm")):
    metric = MeanAveragePrecision(
        iou_type=metric_type,
        class_metrics=False,
        extended_summary=False
    )
    for image_name, image, annotation in dataset:
        detections: sv.Detection = prediction_callback(image)
        ground_truth = {
            "boxes":  Tensor(annotation.xyxy),
            "masks":  BoolTensor([] if annotation.mask is None else annotation.mask),
            "labels": IntTensor(annotation.class_id)
        }

        # print(detections.mask)

        if detections.xyxy is not None and detections.mask is None:
            print(detections.xyxy)

        predictions = {
            "boxes": Tensor(detections.xyxy),
            "masks": BoolTensor([] if detections.mask is None else detections.mask),
            "labels": IntTensor(detections.class_id),
            "scores": Tensor(detections.confidence),
        }
        metric.update([predictions], [ground_truth])
    result = metric.compute()
    print(result)
    return result


if __name__ == '__main__':


    # dataset = sv.DetectionDataset.from_yolo(
    #     "../datasets/trashseg10zerowastecropped/train/images",
    #     "../datasets/trashseg10zerowastecropped/train/labels",
    #     "../datasets/trashseg10zerowastecropped/data.yaml")






    #plt.savefig('barplot_latex.pgf', format='pgf')#, #bbox_inches='tight')

    #print(results)

    # font = {'family' : 'normal',
    #         'weight' : 'bold',
    #         'size'   : 22}

    # matplotlib.rc('font', **font)

    image_path = "../datasets/trashseg11trashboxcropped/train/images/plastic_9_jpg.rf.154a55e075332c8f762b8e326ee1a02f.jpg"
    image = cv2.imread(image_path)
    #hsv_min_max = utils.find_optimal_hsv_threshold_interactively(image)
    hsv_min_max = ([0, 0, 100], [179, 255, 255])
    print(hsv_min_max)

    #model_trashboxsegnormal = YOLO("../models/trashboxsegment/trashboxsegnoaug/weights/best.pt")
    # model_trashboxsegtransformed = YOLO("../models/trashboxsegment/trashboxsegaugsplitted/weights/best.pt")

    model_garbageclassseg = YOLO("../models/garbageclassificationsegment_augmented224yolov8n/weights/best.pt")
    # model_zerowaste = YOLO("../models/zerowasteaug160yolov8n-seg/weights/best.pt")

    model_classifier_trashbox = YOLO("../models/trashboxclassify/trashboxclassifyyolov8n100/weights/best.pt")
    # model_classifier_trashnet = YOLO("../models/trashnet_classify/trashnet-splitted-classifier/weights/best.pt")

    # conf_matrix, names = test_accuracy_yolo(model_classifier_trashbox, "../datasets/trashboxidealcutouts",
    #                     plot_random=True, n_samples = 8)

    # confmatrix_plot("confmatrixnormalized.pdf", conf_matrix, names)

    #confmatrix_plot("confmatrixnormalized.pdf", conf_matrix_normalized, names)

    # dataset_cropped_zerowaste = sv.DetectionDataset.from_yolo(
    #     "../datasets/trashseg10zerowastecropped/train/images",
    #     "../datasets/trashseg10zerowastecropped/train/labels",
    #     "../datasets/trashseg10zerowastecropped/data.yaml")

    # dataset_cropped_garbageclassseg = sv.DetectionDataset.from_yolo(
    #     "../datasets/trashseg12garbageclasssegcropped/train/images",
    #     "../datasets/trashseg12garbageclasssegcropped/train/labels",
    #     "../datasets/trashseg12garbageclasssegcropped/data.yaml")

    dataset_cropped_trashboxseg = sv.DetectionDataset.from_yolo(
        "../datasets/trashseg11trashboxcropped/train/images",
        "../datasets/trashseg11trashboxcropped/train/labels",
        "../datasets/trashseg11trashboxcropped/data.yaml")




    # it = PerfectDetections(dataset_cropped_trashboxseg)
    # detect_trashbox2stageperfect = partial(
    #     combined_seg,
    #     partial(display_det, it ),
    #     partial(yolov8_classify, model_classifier_trashbox)
    # )
    # evaldataset("./results/segment_trashbox2stageperfect", dataset_cropped_trashboxseg, detect_trashbox2stageperfect)



    # dataset_cropped_object = sv.DetectionDataset.from_yolo(
    #     "../datasets/trashseg13objectcropped/train/images",
    #     "../datasets/trashseg13objectcropped/train/labels",
    #     "../datasets/trashseg13objectcropped/data.yaml")

    # detect_zerowaste = partial(yolov8_detect,
    #                            model_zerowaste,
    #                            class_mapping=get_class_mapping(
    #                                model_zerowaste.names,
    #                                dataset_cropped_zerowaste.classes))

    # detect_garbageclassseg = partial(yolov8_detect,
    #                            model_garbageclassseg,
    #                            class_mapping=get_class_mapping(
    #                                model_garbageclassseg.names,
    #                                dataset_cropped_garbageclassseg.classes))

    # detect_trashboxsegnormal = partial(yolov8_detect,
    #                                    model_trashboxsegnormal,
    #                                    class_mapping=get_class_mapping(
    #                                        model_trashboxsegnormal.names,
    #                                        dataset_cropped_trashboxseg.classes))


    # detect_trashboxsegtransformed = partial(yolov8_detect,
    #                                    model_trashboxsegtransformed,
    #                                    class_mapping=get_class_mapping(
    #                                        model_trashboxsegtransformed.names,
    #                                        dataset_cropped_trashboxseg.classes))


    # detect_object_zerowaste = partial(yolov8_detect,
    #                            model_zerowaste,
    #                            class_mapping=get_single_class_mapping(model_zerowaste.names))


    detect_object_garbageclassseg = partial(yolov8_detect,
                                            model_garbageclassseg,
                                            class_mapping=get_single_class_mapping(model_garbageclassseg.names))

    # detect_object_trashboxsegnormal = partial(yolov8_detect,
    #                                           model_trashboxsegnormal,
    #                                           class_mapping=get_single_class_mapping(model_trashboxsegnormal.names))

    # detect_object_trashboxsegtransformed = partial(yolov8_detect,
    #                                           model_trashboxsegtransformed,
    #                                           class_mapping=get_single_class_mapping(
    #                                               model_trashboxsegtransformed.names))

    detect_object_hsv_colorthresh = partial(color_thresh_detect, color_hsv_min_max=hsv_min_max)


    detect_trashbox2stagecolorthresh = partial(
        combined_seg,
        detect_object_hsv_colorthresh,
        partial(yolov8_classify, model_classifier_trashbox))

    detect_trashbox2stagegarbageclass = partial(
        combined_seg,
        detect_object_garbageclassseg,
        partial(yolov8_classify, model_classifier_trashbox)
    )


    # evaldataset("./results/detect_object_hsvcolorthresh", dataset_cropped_object, detect_object_hsv_colorthresh)
    # evaldataset("./results/detect_object_zerowaste", dataset_cropped_object, detect_object_zerowaste)
    # evaldataset("./results/detect_object_garbageclassseg", dataset_cropped_object, detect_object_garbageclassseg)
    # evaldataset("./results/detect_object_trashboxsegnormal", dataset_cropped_object, detect_object_trashboxsegnormal)
    # evaldataset("./results/detect_object_trashboxsegtransformed", dataset_cropped_object, detect_object_trashboxsegtransformed)

    # evaldataset("./results/segment_zerowaste", dataset_cropped_zerowaste, detect_zerowaste)
    # evaldataset("./results/segment_garbageclassseg", dataset_cropped_garbageclassseg, detect_garbageclassseg)

    # evaldataset("./results/segment_trashboxsegnormal", dataset_cropped_trashboxseg, detect_trashboxsegnormal)
    # evaldataset("./results/segment_trashboxsegtransformed", dataset_cropped_trashboxseg, detect_trashboxsegtransformed)
    evaldataset("./results/segment_trashbox2stagecolorthresh", dataset_cropped_trashboxseg, detect_trashbox2stagecolorthresh)


    evaldataset("./results/segment_trashbox2stagegarbageclass", dataset_cropped_trashboxseg,
                 detect_trashbox2stagegarbageclass)



