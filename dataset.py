import cv2
import numpy as np
import random
import re
import math
from pathlib import Path
import os
import json
import yaml
import shutil
import glob

def get_rotation_matrix(center, angle_degrees):
    """returns 3x3 matrix for further general processing with non affine transformations"""
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotation_matrix = np.row_stack([rotation_matrix, [0, 0, 1]])
    return rotation_matrix

def transform_contour(contour, matrix):
    """applies a transformation matrix on a contour"""
    contour = contour.reshape(-1, 1, 2).astype(np.float32)
    contour = cv2.perspectiveTransform(contour, matrix)
    contour = contour.astype(np.int32)
    return contour

def transform_image(image, matrix):
    """applies a transformation matrix on an image"""
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, matrix, (width, height))

def get_quad(center, length):
    """ returns a numpy array with 4 point coordinates"""
    return np.array([
            center + np.array([-length, -length]),
            center + np.array([length, -length]),
            center + np.array([-length, length]),
            center + np.array([length, length])
    ], dtype=np.float32)

def blend_images_with_mask(foreground_image, background_image, foreground_mask):
    """ blends foreground and background image together according to the foreground mask"""
    # the values should be
    # between 0 and 1
    foreground_mask = foreground_mask.astype(float) / 255
    background_mask = np.ones(foreground_mask.shape[:2]) - foreground_mask

    # to multiply with the images
    # we need three channels
    foreground_mask = np.dstack(
        (foreground_mask,
         foreground_mask,
         foreground_mask))

    background_mask = np.dstack(
        (background_mask,
         background_mask,
         background_mask))

    # now we can blend them together
    image = (foreground_mask * foreground_image + background_mask * background_image)
    image /= (foreground_mask + background_mask)
    image = image.astype(dtype=np.uint8)
    return image


def pack_circles_with_resizing(radii, plane_width, plane_height, scaling_factor=0.83, colliding_threshold=8):
    """
    uniform sampling circle packing algorithm returns resized radii and
    randomly placed coordinates in a tuple of arrays (radii, centers)
    """
    radii = radii.copy()
    coords = []

    num_points= len(radii)
    current_radius_index = 0
    colliding_count = 0

    marker_coords = [
        [0, 0],
        [0, plane_height],
        [plane_width, 0],
        [plane_width, plane_height]
    ]
    marker_radius = 0.104 * plane_width

    while len(coords) < num_points:

        x = random.randint(0, plane_width)
        y = random.randint(0, plane_height)
        circle_coord = [x, y]

        is_colliding = False

        # check collisions
        # with markers
        for marker_coord in marker_coords:
            circle_distance = math.dist(marker_coord, circle_coord)
            if circle_distance < radii[current_radius_index] + marker_radius:
                is_colliding = True
                #print("marker colliding")
                break

        # check other collisions
        if not is_colliding:
            for c in range(0, len(coords)):
                circle_distance = math.dist(coords[c], circle_coord)
                if circle_distance <= radii[current_radius_index] + radii[c]:
                    is_colliding = True
                    #print("colliding")
                    break

        if is_colliding:
            colliding_count += 1
        else:
            coords.append([x, y])
            current_radius_index += 1
            colliding_count = 0

        # if the circle is not really
        # fitting anywhere we might
        # scale the largest circle down
        # to get more space
        if colliding_count > colliding_threshold:
            max_radius_index = np.argmax(radii)
            radii[max_radius_index] *= scaling_factor
            colliding_count = 0
    return (radii, coords)


def labelled_contours_to_yolov8_string(image_width, image_height, labelled_contours):
    """
    takes labelled contours and normalizes them to create a string in yolov8 format
    """
    label_strings = []
    for labelled_contour in labelled_contours:
        label, contour = labelled_contour
        # the yolov8 annotation format
        # needs for each detection a label
        # followed by x y normalized coordinates:
        # label x y x y x y x y
        normalized_contour = contour / [image_width, image_height]
        normalized_contour = normalized_contour.flatten()
        contour_str = ' '.join(map(str, normalized_contour))
        label_strings.append(f"{label} {contour_str}")
    label_string = '\n'.join(label_strings)
    return label_string

def yolov8_string_to_labelled_contours(image_width, image_height, string):
    """takes a string in annotation format and returns a labelled contour"""
    labelledContours = []
    lines = string.splitlines()
    for line in lines:
        label, *contour = line.split(' ')
        contour = np.asarray(contour, dtype=np.float16)
        label = int(label)
        contour = contour.reshape(-1,2)
        contour *= [image_width, image_height]
        contour = np.asarray(contour, dtype=np.int32)
        labelledContours.append((label, contour))
    return labelledContours


def save_image_segmentation(imagesegmentation, image_filename, label_filename):
    """
    saves the image and corresponding annotations to the
    specified image and label path in yolov8 format
    """
    image, labelledcontours = imagesegmentation
    image_height, image_width = image.shape[:2]
    os.makedirs(os.path.dirname(image_filename), exist_ok=True)
    os.makedirs(os.path.dirname(label_filename), exist_ok=True)
    annotation_content = labelled_contours_to_yolov8_string(image_width, image_height, labelledcontours)
    with open(label_filename, "w") as f:
        f.write(annotation_content)
    cv2.imwrite(image_filename, image)


def load_image_segmentation(image_filename, label_filename):
    """
    loads an image and all the labelled contours
    the label_filename should be a txt file in yolov8 format
    """
    image = cv2.imread(image_filename)
    with open(label_filename, 'r') as f:
        string = f.read()
    image_height, image_width = image.shape[:2]
    labelledContours = yolov8_string_to_labelled_contours(image_width, image_height, string)
    return (image, labelledContours)


def scale_image_segmentation(imagesegmentation, dst_width, dst_height):
    """ scales the image and all the labelled contours """
    image, labelledcontours = imagesegmentation
    src_height, src_width = image.shape[:2]

    # maybe we don't have to do anything
    if src_height == dst_height and src_width == dst_width:
        return imagesegmentation

    # we need to scale the image
    # as well as the contours
    matrix = cv2.getPerspectiveTransform(
        np.array([[0, 0], [src_width, 0], [src_width, src_height], [0, src_height]], dtype=np.float32),
        np.array([[0, 0], [dst_width, 0], [dst_width, dst_height], [0, dst_height]], dtype=np.float32))

    result_labelled_contours = []
    for label, contour in labelledcontours:
        contour =  transform_contour(contour, matrix)
        result_labelled_contours.append((label, contour))

    image = transform_image(image, matrix)
    result_image = image[0:dst_height, 0:dst_width]
    return (result_image, result_labelled_contours)


def image_path_to_label_path(image_path):
    """
    for every image in the images folder
    there should be txt file in the labels directory
    """
    label_path = "/labels/".join(image_path.rsplit("/images/", 1))
    label_path = ".txt".join(label_path.rsplit(".jpg"))
    return label_path

def get_random_color():
    """ returns a tuple with random RGB values """
    return tuple(np.random.random(size=3) * 255)


def object_mask_to_foreground_mask(object_mask):
    """
    modifies the mask of an object for image blending
    """
    mask = object_mask.copy()
    kernel = np.ones((8,8),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    return mask


def image_segmentations_to_labels_masks_images(image_segmentations):
    """
    unfolds the image and labelled contours into seperate
    labels images and masks for further processing
    """
    labels = []
    images = []
    masks = []
    for image_segmentation in image_segmentations:
        image, labelledcontours = image_segmentation
        for label, contour in labelledcontours:
            imgcpy = image.copy()
            mask = np.zeros(imgcpy.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            masks.append(mask)
            images.append(imgcpy)
            labels.append(label)
    return (labels, masks, images)


def apply_transform_on_labels_masks_images(labelsmasksimages, matrices):
    """ applies a transformation matrix on masks and images """
    labels, masks, images = labelsmasksimages
    for i in range(0, len(masks)):
        matrix = matrices[i]
        masks[i] = transform_image(masks[i], matrix)
        images[i] = transform_image(images[i], matrix)
    return (labels, masks, images)


def labels_masks_images_to_image_segmentations(labelsmasksimages, background, objectMask2ForegroundMask):
    labels, masks, images = labelsmasksimages

    # it is very unlikely
    # but maybe some masks
    # don't contain visible
    # contours anymore
    # so we have to filter
    # those out
    object_contours = []
    object_masks = []
    object_labels = []
    object_images = []

    for i in range(0, len(masks)):
        # we can assume that there is only
        # one contour in the new image
        mask = masks[i]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("filtered contour")
            continue
        contour = contours[0]
        object_contours.append(contour)
        object_masks.append(mask)
        object_labels.append(labels[i])
        object_images.append(images[i])

    # blend images together
    result_image = background.copy()
    for i in range(0, len(object_masks)):
        foreground_mask = objectMask2ForegroundMask(object_masks[i])
        result_image = blend_images_with_mask(object_images[i], result_image, foreground_mask)

    result_labelled_contours = list(zip(object_labels, object_contours))
    return (result_image, result_labelled_contours)


def radii_centers_to_transformation_matrices(src_radii, src_centers, image_width, image_height):
    matrices = []
    dst_radii, dst_centers = pack_circles_with_resizing(
        src_radii,
        image_width,
        image_height
    )
    for i in range(0, len(dst_centers)):
        src = get_quad(src_centers[i], src_radii[i])
        dst = get_quad(dst_centers[i], dst_radii[i])
        movematrix = cv2.getPerspectiveTransform(src, dst)
        angle_degrees = random.randint(0, 359)
        rotationmatrix = get_rotation_matrix(dst_centers[i], angle_degrees)
        matrices.append(np.matmul(rotationmatrix, movematrix))
    return matrices


def image_segmentations_to_radii_centers(imagesegmentations):
    centers = []
    radii = []
    for image, labelledcontours in imagesegmentations:
        for label, contour in labelledcontours:
            center, radius = cv2.minEnclosingCircle(contour)
            centers.append(center)
            radii.append(radius)
    return (radii, centers)


def visualize_image_segmentation(image_segmentation, opacity=0.8):
    img, labelled_contours = image_segmentation
    image = img.copy()
    overlay = image.copy()
    for i, (label, contour) in enumerate(labelled_contours):
        cv2.drawContours(image, [contour], 0, get_random_color(), 3)
        font = cv2.FONT_HERSHEY_DUPLEX
        cx, cy, w, h = cv2.boundingRect(contour)
        cx += int(0.5 * w)
        cy += int(0.5 * h)
        cv2.putText(image, str(label), (cx, cy), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    image = cv2.addWeighted(image, opacity, overlay, 1.0 - opacity, 0)
    return image


# solution found here:
# https://stackoverflow.com/questions/62941378/how-to-sort-glob-glob-numerically
file_pattern = re.compile(r'.*?(\d+).*?')
def get_numerical_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def get_all_images_in_numerical_order(dirpath):
    dirpath = os.path.abspath(dirpath)
    return sorted(glob.glob(f"{dirpath}/**/*.jpg", recursive=True), key=get_numerical_order)


def load_or_init_checkpoint(checkpoint_path, data):
    try:
        f = open(checkpoint_path, "r")
        checkpoint = json.load(f)
        f.close()
    except:
        print(f"couldnt open {checkpoint_path}")
        checkpoint = data
    return checkpoint

def save_checkpoint(checkpoint_path, checkpoint):
    try:
        f = open(checkpoint_path, "w")
        json.dump(checkpoint, f)
        f.close()
    except IOError as e:
        print(e)

def stitch_image_segmentations_together(unscaled_imagesegmentations, background):
    image_height, image_width = background.shape[:2]
    image_segmentations = []
    for imgseg in unscaled_imagesegmentations:
        image_segmentations.append(scale_image_segmentation(imgseg, image_width, image_height))

    labelsmasksimages = image_segmentations_to_labels_masks_images(image_segmentations)
    src_radii, src_centers = image_segmentations_to_radii_centers(image_segmentations)
    matrices = radii_centers_to_transformation_matrices(src_radii, src_centers, image_width, image_height)
    apply_transform_on_labels_masks_images(labelsmasksimages, matrices)
    newimagesegmentation = labels_masks_images_to_image_segmentations(labelsmasksimages, background,
                                                                      object_mask_to_foreground_mask)
    return newimagesegmentation

def image_paths_to_image_segmentations(image_paths, only_biggest_contour=True):
    image_segmentations = []
    for image_path in image_paths:
        annotation_path = image_path_to_label_path(image_path)
        try:
            image_segmentation = load_image_segmentation(image_path, annotation_path)
        except IOError as e:
            print(f"warning: propably missing annotation for image {image_path}. annotation_path: {annotation_path}")
            continue
        if only_biggest_contour:
            image, labelledcontours = image_segmentation
            if len(labelledcontours) == 0:
                continue
            labelledcontours = [sorted(labelledcontours, key=lambda labelcontour: cv2.contourArea(labelcontour[1]))[0]]
            image_segmentation = (image, labelledcontours)
        image_segmentations.append(image_segmentation)
    return image_segmentations


def get_simple_background_provider(directory):
    image_paths = glob.glob(f"{os.path.abspath(directory)}/**/*.jpg", recursive=True)
    def get_random_background():
        i = random.randrange(len(image_paths))
        image_path = image_paths[i]
        image = cv2.imread(image_path)
        contrast = random.uniform(0, 3) #5. # Contrast control ( 0 to 127)
        brightness = random.uniform(0, 3) #2. # Brightness control (0-100)
        image = cv2.addWeighted(image, contrast, image, 0, brightness)
        return image
    return get_random_background


def seg_augment(src_dir, dst_dir, get_background, max_object_count, image_prefix="", seed=3):
    """generates a new dataset with just augmented image segmentations"""
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    os.makedirs(f"{dst_dir}/labels", exist_ok=True)
    os.makedirs(f"{dst_dir}/images", exist_ok=True)

    checkpoint_path = f"{dst_dir}/checkpoint.json"
    checkpoint = load_or_init_checkpoint(checkpoint_path, {"seed": seed, "it":0, "max_object_count": max_object_count})
    max_object_count = checkpoint["max_object_count"]

    image_paths = get_all_images_in_numerical_order(src_dir)
    random.seed(checkpoint["seed"])
    random.shuffle(image_paths)
    selections = []
    i = 0
    while i < len(image_paths):
        num_selected_images = random.randint(1, max_object_count)
        selections.append((i, i+num_selected_images))
        i += num_selected_images

    print(f"{len(selections)} images to generate")
    while checkpoint["it"] < len(selections):
        start, end = selections[checkpoint["it"]]
        selected_image_paths = image_paths[start:end]
        background = get_background()
        image_segmentations = image_paths_to_image_segmentations(selected_image_paths)
        newimagesegmentation = stitch_image_segmentations_together(image_segmentations, background)

        output_image_path = f"{dst_dir}/images/{image_prefix}image_{start}_{end}.jpg"
        output_label_path = image_path_to_label_path(output_image_path)
        save_image_segmentation(newimagesegmentation, output_image_path, output_label_path)
        checkpoint["it"] += 1
        save_checkpoint(checkpoint_path, checkpoint)
    print("done")


def seg_remap(src_dir, dst_dir, remap, remove_backgrounds=False):
    """outputs the same segmentation dataset with remapped labels. all labels not specified are dropped."""
    # this calculation is done
    # because we want to continue
    # where we left of
    # in case this function was interrupted
    relative_image_paths = [ os.path.relpath(p, src_dir) for p in get_all_images_in_numerical_order(src_dir)]
    relative_image_paths_processed = [os.path.relpath(p, dst_dir) for p in get_all_images_in_numerical_order(dst_dir)]
    relative_image_paths_not_processed = list(set(relative_image_paths) - set(relative_image_paths_processed))

    for relative_image_path in relative_image_paths_not_processed:
        image_path = f"{src_dir}/{relative_image_path}"
        label_path = image_path_to_label_path(image_path)
        image_segmentation = load_image_segmentation(image_path, label_path)
        image, labelledcontours = image_segmentation

        relabelledcontours = []
        skip_image = False
        for label, contour in labelledcontours:
            if label not in remap:
                skip_image = True
                break
            newlabel = remap[label]
            relabelledcontours.append((newlabel, contour))

        if remove_backgrounds and len(labelledcontours) == 0:
            skip_image = True

        if skip_image:
            continue

        new_image_segmentation = (image, relabelledcontours)

        output_image_path = image_path.replace(src_dir, dst_dir)
        output_label_path = image_path_to_label_path(output_image_path)
        save_image_segmentation(new_image_segmentation, output_image_path, output_label_path)

def get_yaml_name_remap(yaml_file, dstName2Category):
    with open(yaml_file, "r") as f:
        try:
            info = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            return None

    srcName2Category = {}
    for i, name in enumerate(info["names"]):
        srcName2Category[name] = i

    category2Category = {}
    for name in dstName2Category:
        category2Category[srcName2Category[name]] = dstName2Category[name]
    return category2Category


def seg_name_remap(src_dir, dst_dir, name_category_map, remove_backgrounds=False):
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    data_yaml_path = f"{src_dir}/data.yaml"
    category_remap = get_yaml_name_remap(data_yaml_path, name_category_map)
    seg_remap(src_dir, dst_dir, category_remap, remove_backgrounds)


def combine_and_split_segs(seg_paths, output_path, weights=(0.7, 0.2, 0.1)):
    output_path = os.path.abspath(output_path)
    possible_subdirs = ["train", "val", "test"]

    for subdir in possible_subdirs:
        output_label_dir = f"{output_path}/{subdir}/labels/"
        output_images_dir = f"{output_path}/{subdir}/images/"
        os.makedirs(output_label_dir, exist_ok=True)
        os.makedirs(output_images_dir, exist_ok=True)

    seg_paths = list(map(os.path.abspath, seg_paths))
    for seg_path in seg_paths:
        basename = os.path.basename(seg_path)
        image_paths = get_all_images_in_numerical_order(seg_path)

        # choose a folder for each image
        image_subdirs = random.choices(possible_subdirs, weights=weights, k=len(image_paths))

        for i in range(0, len(image_paths)):
            input_image_path = image_paths[i]
            input_label_path = image_path_to_label_path(input_image_path)

            output_image_path = f"{output_path}/{image_subdirs[i]}/images/{basename}_image_{i}.jpg"
            output_label_path = image_path_to_label_path(output_image_path)

            shutil.copyfile(input_label_path, output_label_path)
            shutil.copyfile(input_image_path, output_image_path)


def save_standard_yaml(dst_yaml_path, category_label_map):
    # sort by label numbers
    category_label_map = dict(sorted(category_label_map.items(), key=lambda item: item[1]))

    os.makedirs(os.path.dirname(dst_yaml_path), exist_ok=True)

    info = {}
    info["names"] = [key for key in category_label_map]

    with open(dst_yaml_path, "w") as f:
        yaml.dump(info, f, default_flow_style=False)


def combine_segs(datasetname2names, output_path, name2intcategory, weights, get_background=None, max_object_count=3, seed=3):
    output_path = os.path.abspath(output_path)
    save_standard_yaml(f"{output_path}/data.yaml", name2intcategory)

    dataset_name_category_mapping = {}
    for datasetname in datasetname2names:
        name2categorymap = {}
        for datasetcategoryname in datasetname2names[datasetname]:
            name2categorymap[datasetcategoryname] = name2intcategory[datasetname2names[datasetname][datasetcategoryname]]
        dataset_name_category_mapping[datasetname] = name2categorymap


    remapped_dataset_names = {}
    for datasetname in dataset_name_category_mapping:
        remapped_dataset_names[datasetname] = f"{output_path}/remapped_{os.path.basename(datasetname)}"

    augmented_dataset_names = {}
    for datasetname in remapped_dataset_names:
        augmented_dataset_names[datasetname] = f"{output_path}/augmented_{os.path.basename(datasetname)}"

    for dataset_path in remapped_dataset_names:
         seg_name_remap(
             dataset_path,
             remapped_dataset_names[dataset_path],
             dataset_name_category_mapping[dataset_path])

    generated_datasets = list(remapped_dataset_names.values())

    if get_background is not None:
        for dataset_path in augmented_dataset_names:
            seg_augment(dataset_path,
                        augmented_dataset_names[dataset_path],
                        get_background,
                        max_object_count,
                        image_prefix=f"{os.path.basename(dataset_path)}_",
                        seed=seed)
            generated_datasets.append(augmented_dataset_names[dataset_path])

    combine_and_split_segs(generated_datasets, output_path)


def cutout_quad(image, quad):
    image_height, image_width = image.shape[:2]
    p1 = quad[0]
    p2 = quad[1]
    p3 = quad[2]
    y1 = int(p1[1])
    y2 = int(p3[1])
    x1 = int(p1[0])
    x2 = int(p2[0])
    y1 = max(0, y1)
    y2 = min(image_height, y2)
    x1 = max(0, x1)
    x2 = min(image_width, x2)
    return image[y1:y2, x1:x2]


def cutout_single_image_segmentation(image, masks, labels):
    imagesegmentations = []
    for label, mask in zip(labels, masks):


        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        contour = contours[0]


        kernel = np.ones((10,10),np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations = 3)
        contours_dilated, _ = cv2.findContours(mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_dilated) == 0:
            continue
        contour_dilated = contours_dilated[0]

        x, y, w, h = cv2.boundingRect(contour_dilated)

        if cv2.contourArea(contour) < 100:
            continue

        src = np.array([
            [x, y],
            [x+w, y],
            [x+w, y+h],
            [x, y+h]], dtype=np.float32)

        image_cutout = cutout_quad(image, src)
        mask_cutout =  cutout_quad(mask, src)

        contours, _ = cv2.findContours(mask_cutout, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        contour = contours[0]
        imagesegmentations.append((image_cutout, [(label, contour)]))
    return imagesegmentations



############ GROUNDING DINO + SAM ##########

from typing import List
import torch
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

def enhanceClassName(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)



def cls2seg(grounding_dino_model,
            sam_predictor,
            src_dir,
            dst_dir,
            category_to_label_map,
            box_threshold = 0.35,
            text_threshold = 0.25):


    # normalize paths
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    dst_yaml_path = f"{dst_dir}/data.yaml"

    save_standard_yaml(dst_yaml_path, category_to_label_map)

    image_paths = get_all_images_in_numerical_order(src_dir)

    dst_checkpoint_path = f"{dst_dir}/checkpoint.txt"
    try:
        f = open(dst_checkpoint_path, "r")
        last_image_path = f.read()
        f.close()
        skip = True
    except IOError:
        skip = False

    for image_path in image_paths:
        if skip:
            if image_path == last_image_path:
                skip = False
            continue

        splitted_path = os.path.normpath(image_path).split(os.sep)

        image_name = splitted_path[-1]
        category = splitted_path[-2]
        train_val_or_test = splitted_path[-3]

        dst_image_name = f"{dst_dir}/{train_val_or_test}/images/{image_name}"
        dst_label_name = image_path_to_label_path(dst_image_name)

        # check if it is already processed
        if os.path.exists(dst_image_name):
            continue

        # skip those which are not specified
        if category not in category_to_label_map:
            continue

        label = category_to_label_map[category]
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_height, image_width = image.shape[:2]

        # detect objects
        # with grounding dino
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhanceClassName(class_names=[category]),
                 box_threshold=box_threshold,
                 text_threshold=text_threshold
          )

        with open(dst_checkpoint_path, "w") as f:
            print(f"{image_path}")
            f.write(image_path)

        if len(detections) == 0:
            continue

        # convert detections
        # to masks with Sam
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        masks = []
        confs = []
        areas = []
        for xyxy, mask, confidence, class_id, _ in detections:
            masks.append(mask.astype(dtype='uint8') * 255)
            confs.append(confidence)
            areas.append(np.count_nonzero(mask))

        mask_index = np.argmax(confs)
        mask = masks[mask_index]

        imagesegmentations = cutout_single_image_segmentation(image, [mask], [label])
        if len(imagesegmentations) == 0:
            continue

        save_image_segmentation(imagesegmentations[0], dst_image_name, dst_label_name)
    return



def load_segmentation_ground_truth(image_dimension, annotation_path, isbox):
    """loads contours / masks / labels and boxes from an yolov8 annotation txt file"""
    with open(annotation_path, 'r') as f:
        lines = f.read().splitlines()

    image_height, image_width = image_dimension

    # detected boxes with format
    # [xmin, ymin, xmax, ymax]
    # in absolute image coordinates
    true_boxes = []

    # boolean mask of
    # the image per category
    true_masks = []
    true_labels = []
    true_names = []
    true_contours = []

    # extract truth values
    # from annotation path
    for line in lines:

        # extract info
        label, *contour = line.split(' ')
        contour = np.asarray(contour, dtype=np.float16)
        label = int(label)

        # convert to
        # correct format
        if isbox:
            cx = contour[0] * image_width
            cy = contour[1] * image_height
            w  = contour[2] * image_width
            h  = contour[3] * image_height
            xoffset = w/2
            yoffset = h/2
            contour = np.array([
                [cx-xoffset, cy-yoffset],
                [cx+xoffset, cy-yoffset],
                [cx+xoffset, cy+yoffset],
                [cx-xoffset, cy+yoffset],
            ])
            box = [cx-xoffset, cy-yoffset, cx+xoffset, cy+yoffset]

        else:
            contour = contour.reshape(-1,2)
            contour *= [image_width, image_height]
            contour = np.asarray(contour, dtype=np.int32)
            box_x,box_y,box_w,box_h = cv2.boundingRect(contour)
            box = [box_x, box_y, box_x + box_w, box_y + box_h]

        mask = np.zeros((image_height, image_width), np.uint8)
        cv2.drawContours(mask, [contour.astype('int')], 0, 255, -1)

        true_contours.append(contour)
        true_masks.append(mask)
        true_labels.append(label)
        true_boxes.append(box)

    return {
        "boxes": true_boxes,
        "labels": true_labels,
        "masks": true_masks,
        "contours": true_contours
    }


def objectdet2seg(
    sam_predictor,
    object_detection_directory,
    output_segmentation_directory,
    box_threshold = 0.35,
    text_threshold = 0.25,
    standard_size=[224, 224]):
  """ converts an object detection dataset into a segmentation dataset using segment anything model"""

  object_detection_directory = os.path.abspath(object_detection_directory)
  output_segmentation_directory = os.path.abspath(output_segmentation_directory)

  # we expect a data.yaml file in the directory
  # containing some nice configuration data
  expected_data_yaml_file = f"{object_detection_directory}/data.yaml"
  with open(expected_data_yaml_file, "r") as f:
    try:
      info = yaml.safe_load(f)
    except yaml.YAMLError as exc:
      print(exc)
      return

  output_yaml_file = f"{output_segmentation_directory}/data.yaml"
  with open(output_yaml_file, "w") as f:
    yaml.dump(info, f, default_flow_style=False)

  # the category names and root directory
  # are specified in the yaml file
  categories = info["names"]

  if 'path' in info:
    dataset_root = f"{object_detection_directory}/{info['path']}"
  else:
    dataset_root = object_detection_directory

  # these directories might
  # have been set in the data.yaml file
  possible_directories = ["train", "val", "test"]

  # now we can start converting
  # everything
  for possible_dir in possible_directories:
    input_images_directory = f"{dataset_root}/{possible_dir}/images"
    input_labels_directory = "/labels".join(input_images_directory.rsplit("/images", 1))

    print(input_images_directory)
    print(input_labels_directory)

    if os.path.isdir(input_images_directory) and os.path.isdir(input_labels_directory):

      output_images_directory = f"{output_segmentation_directory}/{possible_dir}/images/"
      output_labels_directory = f"{output_segmentation_directory}/{possible_dir}/labels/"
      print(f"output {output_images_directory} {output_labels_directory}")

      # create needed directories
      # if they don't exist yet
      os.makedirs(output_images_directory, exist_ok=True)
      os.makedirs(output_labels_directory, exist_ok=True)

      # maybe we have already done some
      # segmentations saved in the folder
      # so we don't have to redo everything
      existing_basenames = [os.path.basename(x) for x in glob.glob(f"{output_images_directory}/*.jpg")]
      print(existing_basenames)

      # do only segmentations for
      # images that have not yet been written
      # to the destination (even on second call of the function)
      # because this function might interrupt due to time / resource limits
      input_basenames = [os.path.basename(x) for x in glob.glob(f"{input_images_directory}/*.jpg")]
      not_processed_images_basenames = sorted(list(set(input_basenames) - set(existing_basenames)))

      print(f"not processed: {not_processed_images_basenames}")

      for input_image_basename in not_processed_images_basenames:

        # each input image will have
        # a corresponding txt file
        label_basename = os.path.splitext(input_image_basename)[0]  + ".txt"

        # now define the full output path
        # for each file
        output_image_path = f"{output_images_directory}/{input_image_basename}"
        output_label_path = f"{output_labels_directory}/{label_basename}"
        input_image_path = f"{input_images_directory}/{input_image_basename}"
        input_label_path = f"{input_labels_directory}/{label_basename}"

        print(f"processing {input_image_path}")

        # load image and
        # ensure that it is valid
        image = cv2.imread(input_image_path)
        if image is None:
          continue

        # load box annotations from
        # the label file
        results = load_segmentation_ground_truth(image.shape[:2], input_label_path, isbox=True)

        boxes = np.array(results["boxes"])
        labels = results["labels"]


        if len(boxes) == 0:
          print("no boxes")

        # convert bounding boxes
        # to masks with Sam
        masks = segment(
          sam_predictor=sam_predictor,
          image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
          xyxy=boxes
        )

        if len(masks) == 0:
          print("no masks")

        # optionally resize the image
        if standard_size is not None:
          image = cv2.resize(image, standard_size, interpolation = cv2.INTER_AREA)

        label_content = []

        image_height, image_width = image.shape[:2]
        for mask, class_id in zip(masks, labels):

            mask = mask.astype(dtype='uint8')
            mask *= 255

            # if the image was resized
            # the masks need also be resized
            if standard_size is not None:
              mask = cv2.resize(mask, standard_size, interpolation=cv2.INTER_AREA)

            # now we need to convert the mask
            # into a contour to save the result
            # in the output label file
            contours, hierarchy  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours is None or len(contours) == 0:
              continue

            # I am not expecting more than one contour
            # but I want to make sure to get the right one
            contour = max(contours, key = cv2.contourArea)

            # yolov8 format expects the contour
            # to be normalized
            normalized_contour = contour / [image_width, image_height]
            normalized_contour = normalized_contour.flatten()

            contour_str = f"{class_id} {' '.join(map(str, normalized_contour))}"
            label_content.append(contour_str)

        content = '\n'.join(label_content)
        print(content)
        # save the annotation
        with open(output_label_path, 'w') as f:
          f.write(content)

        # save the file
        cv2.imwrite(output_image_path, image)


def split_cls(input_dir, output_dir, train_frac, val_frac):
    if train_frac + val_frac > 1.00000001:
        print("train val and test fractions must not exceed 1.0")
        return

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    # get any category names
    # inside the input directory
    categories = next(os.walk(input_dir))[1]


    for category in categories:
        image_paths = glob.glob(f"{input_dir}/{category}/*.jpg")
        num_train = int(train_frac * len(image_paths))
        num_val   = int(val_frac * len(image_paths))

        num_test = len(image_paths) - num_train - num_val
        print(f"{category} {num_train} images for training")
        print(f"{category} {num_val} images for validation")
        print(f"{category} {num_test} images for testing")

        train_images = image_paths[0:num_train]
        val_images = image_paths[num_train:(num_train+num_val)]
        test_images = image_paths[(num_train+num_val):]

        for image_path in train_images:
            output_path = image_path.replace(input_dir, f"{output_dir}/train")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copyfile(image_path, output_path)

        for image_path in val_images:
            output_path = image_path.replace(input_dir, f"{output_dir}/val")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copyfile(image_path, output_path)

        for image_path in test_images:
            output_path = image_path.replace(input_dir, f"{output_dir}/test")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copyfile(image_path, output_path)

from datetime import datetime

def yolo2coco(src_dir):
    src_dir = os.path.abspath(src_dir)
    coco_filename = f"{src_dir}/annotations.json"
    yaml_file = f"{src_dir}/data.yaml"
    with open(yaml_file, "r") as f:
        try:
            info = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            return None

    categories = [
        {
            "supercategory": "object",
            "id": category_id,
            "name": categoryname
        }
        for category_id, categoryname in enumerate(info["names"])
    ]
    date = datetime.today()
    info = {
        "description": "segmentation dataset converted from yolo format",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": date.strftime("%Y"),
        "contributor": "",
        "date_created": date.strftime("%Y/%m/%d")
    }
    licenses = [
        {
            "url": "https://creativecommons.org/licenses/by-nc/4.0/deed.en",
            "id": 1,
            "name": "Attribution-NonCommercial License"
        }
    ]
    images = []
    annotations = []
    image_paths = get_all_images_in_numerical_order(src_dir)
    num_annotation = 0
    for image_id, image_path in enumerate(image_paths):
        label_path = image_path_to_label_path(image_path)
        image, labelledcontours = load_image_segmentation(image_path, label_path)
        image_height, image_width = image.shape[:2]
        coco_image = {
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": image_path,
            "license": 1,
            "coco_url": image_path,
            "date_captured": "0"
        }
        images.append(coco_image)
        for label, contour in labelledcontours:
            x, y, w, h = cv2.boundingRect(contour)
            coco_annotation = {
                "id": num_annotation,
                "image_id": image_id,
                "category_id": label,
                "iscrowd": 0,
                "bbox": [x, y, w, h],
                "segmentation": [list(map(float, contour.flatten()))],
                "area": cv2.contourArea(contour)
            }
            num_annotation += 1
            annotations.append(coco_annotation)
    coco_json = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(coco_filename, "w") as f:
        json.dump(coco_json, f)
