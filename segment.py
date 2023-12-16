import cv2
import numpy as np
from ultralytics import YOLO
import time

def standardize_image(img):
    array_type = img.dtype

    # color balance normalizing
    color_mean = np.mean(img, axis=(0, 1))
    mean_color_mean = np.mean(color_mean)
    img = img[:][:] * mean_color_mean / color_mean

    # color range normalizing
    min_, max_ = np.quantile(img, [0.001, 0.95])
    img = (img - min_) * 256 / (max_ - min_)
    img = np.clip(img, 0, 255)
    img = img.astype(array_type)
    return img

def fill_holes(img):
    """ fills holes that are inside masks """
    im_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img = img | im_floodfill_inv
    return img


def color_threshold_mask(img, color_hsv_min_max = [[0, 70, 0], [179, 255, 255]]):
    frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, tuple(color_hsv_min_max[0]), tuple(color_hsv_min_max[1]))

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # erode workspace markers
    mask[:15, :] = cv2.erode(mask[:15, :], kernel7, iterations=5)
    mask[-15:, :] = cv2.erode(mask[-15:, :], kernel7, iterations=5)
    mask[:, :15] = cv2.erode(mask[:, :15], kernel7, iterations=5)
    mask[:, -15:] = cv2.erode(mask[:, -15:], kernel7, iterations=5)

    mask = fill_holes(mask)
    mask = cv2.dilate(mask, kernel3, iterations=1)
    mask = cv2.erode(mask, kernel5, iterations=1)
    mask = cv2.dilate(mask, kernel11, iterations=1)
    mask = fill_holes(mask)
    mask = cv2.erode(mask, kernel7, iterations=1)
    return mask

def extract_images_for_classification(image, contours):
    images = []
    image_height, image_width = image.shape[:2]
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        square_size = max(w, h)
        center_x = x + w / 2
        center_y = y + h / 2
        #img_cut = np.zeros((square_size, square_size, 3), np.uint8)
        img_cut = image[max(0, int(center_y - square_size / 2)): min(image_height, int(center_y + square_size / 2)),
                              max(0, int(center_x - square_size / 2)): min(image_width, int(center_x + square_size / 2))]
        images.append(img_cut)
    return images



def color_seg(image, color_hsv_min_max = [[0, 70, 0], [179, 255, 255]], only_biggest_contour=False):
    total_start = time.time()
    mask = color_threshold_mask(image, color_hsv_min_max)

    if only_biggest_contour:
        biggest_contour = biggest_contour_finder(mask)
        if biggest_contour is None:
            contours = []
        elif len(biggest_contour) == 0:
            contours = []
        else:
            contours = [biggest_contour]
    else:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    object_masks = []
    for contour in contours:
        obj_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(obj_mask, [contour], 0, 255, -1)
        object_masks.append(obj_mask)

    labels = [0 for i in contours]
    names = ["object" for i in contours]
    scores = [1 for i in contours]
    boxes = [cv2.boundingRect(contour) for contour in contours]
    masks = object_masks

    total_end = time.time()
    total_elapsed = total_end - total_start

    return {
        "labels": labels,
        "names": names,
        "scores": scores,
        "contours": contours,
        "boxes": boxes,
        "masks": object_masks,
        "time_total": total_elapsed
    }



def yolov8_seg(model, image):
    total_start = time.time()
    predictions = model.predict(image, retina_masks=True, conf=0.1)[0]
    if len(predictions) > 0:
        labels   =  list(map(lambda x: int(x.cls),  predictions.boxes))
        names    =  list(map(lambda x: model.names[int(x.cls)],  predictions.boxes))
        scores   =  list(map(lambda x: float(x.conf), predictions.boxes))
        contours =  list(map(lambda x: x.astype('int32'), predictions.masks.xy))
        masks    =  np.asarray(predictions.masks.data.numpy())
        boxes    =  list(map(lambda x: x.xyxy[0].numpy().astype('int32'), predictions.boxes))

    else:
        labels = []
        names = []
        scores = []
        contours = []
        boxes = []
        masks = []

    total_end = time.time()
    total_elapsed = total_end - total_start
    return {
        "labels": labels,
        "names": names,
        "scores": scores,
        "contours": contours,
        "boxes": boxes,
        "masks": masks,
        "time_total": total_elapsed
    }

def yolov8_classify(model, images):
    total_start = time.time()
    predictions = model(images)

    labels = []
    confs = []
    names = []
    for prediction in predictions:
        top1 = prediction.probs.top1
        conf1 = prediction.probs.top1conf
        name = prediction.names[top1]

        labels.append(top1)
        confs.append(float(conf1))
        names.append(prediction.names[top1])

    total_end = time.time()
    total_elapsed = total_end - total_start
    return {
        "labels": labels,
        "confs": confs,
        "names": names
    }


def combined_seg(segment, classify, extract_images_from_contours, image):
    time_start = time.time()
    segmentations = segment(image)
    images_to_classify = extract_images_from_contours(image,
                                                      segmentations["contours"])
    classifications = classify(images_to_classify)
    time_end = time.time()
    time_total = time_end - time_start
    segmentations["time_total"] = time_total
    segmentations["labels"] = classifications["labels"]
    segmentations["names"] = classifications["names"]
    return segmentations













