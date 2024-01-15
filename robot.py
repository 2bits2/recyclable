from pyniryo import *
import json
import segment
from functools import partial
import dataset
import os

from ultralytics import YOLO

def main():

    # initialize robot
    settings = get_settings(is_simulation=True)
    robot = NiryoRobot(settings["ip"])
    robot.calibrate_auto()
    robot.update_tool()
    robot.release_with_tool()

    conveyor_id = ConveyorID.ID_1

    # define which segmentation
    yolomodel = YOLO("../code/resources/garbageclassifcationsegaugyolov8n224.pt")
    infer = partial(segment.yolov8_seg, yolomodel)

    #robot_learn_poses(robot, yolomodel.names, settings)
    #robot_learn_poses(robot, ["paper", "glass", "plastic", "metal", "medical", "e-waste", "cardboard", "trash", "bio", "background", "observation"], settings)
    #robot_learn_poses(robot, ["observation"], settings)

    speed = 0
    time_for_grap = 0.8

    # sorting loop
    while True:
        #robot.run_conveyor(conveyor_id, speed=50)
        #robot.wait(1)
        #robot.stop_conveyor(conveyor_id)

        # move to observation pose
        observation_pose = settings["poses"]["observation"]
        robot.move_pose(PoseObject(**observation_pose))

        # preprocess incoming image
        xy_workspace_offset = (24, 24)
        mtx, dist        = robot.get_camera_intrinsics()
        img_compressed   = robot.get_img_compressed()
        img_raw          = uncompress_image(img_compressed)
        img_undistorted  = undistort_image(img_raw, mtx, dist)
        img_workspace    = extract_img_workspace(img_undistorted, workspace_ratio=1.0)

        if img_workspace is None:
            print("no workspace")
            continue

        # crop the border
        # because there is too much
        # distractions and there can't
        # be any object
        image            = crop_image_border(img_workspace, xy_workspace_offset)

        # infer which objects are present
        inference_result = infer(image)
        names    = inference_result["names"]
        contours = inference_result["contours"]

        viz = dataset.visualize_image_segmentation((image, list(zip(names, contours))))
        cv2.imshow("viz", viz)
        key = cv2.waitKey(0)

        # if no objects are present
        # the conveyor belt can move
        # a little bit and we start all over
        if len(contours) == 0:
            print("no contours")
            #robot.control_conveyor(conveyor_id, True, 20, ConveyorDirection.Forward)
            #robot.control_conveyor(conveyor_id, False, 0, ConveyorDirection.Forward)
            continue

        # get the object
        # that we want to grab
        target_pose_name = names[0]
        contour = contours[0]

        # ensure that the object has
        # a target position
        if target_pose_name not in settings["poses"]:
            input(f"need to learn pose for {target_pose_name}")
            robot_learn_poses(robot, [target_pose_name], settings)


        # get grab information
        cx, cy = get_contour_barycenter(contour)
        angle  = -get_contour_angle(contour)

        # adjust the data
        # because we cropped the
        # image
        xoffset, yoffset = xy_workspace_offset
        cx += xoffset
        cy += yoffset

        # calculate the picking pose
        cx_rel, cy_rel = relative_pos_from_pixels(img_workspace, cx, cy)
        workspace_name = settings["workspace_name"]
        obj_pose = robot.get_target_pose_from_rel(
                workspace_name,
                height_offset=0.0,
                x_rel=cx_rel,
                y_rel=cy_rel,
                yaw_rel=angle)

        # try to pick the object
        # and move it to the correct
        # position
        try:
            robot.pick_from_pose(obj_pose)
            destination = PoseObject(**settings["poses"][target_pose_name])
            intermediate_pose = {
                "x":0.17,
                "y":0.,
                "z":0.35,
                "roll":0.0,
                "pitch":1.57,
                "yaw":0.0
            }
            robot.move_pose(PoseObject(**intermediate_pose))
            robot.place_from_pose(destination)
        except NiryoRobotException as e:
            print(e)
            robot.release_with_tool()
            continue


def get_settings(is_simulation):
    """ depending on the simulation we are in we return different settings :) """
    settings = {}
    if is_simulation:
        settings["ip"] = "127.0.0.1"
        settings["workspace_name"] = "gazebo_1"
    else:
        settings["ip"] = "192.168.0.228"
        #settings["workspace_name"] = "t_workspace"
        settings["workspace_name"] = "t_convbelt"

    settings["pose_filename"] = "./resources/poses.json"

    with open(settings["pose_filename"], "r") as f:
        settings["poses"] = json.load(f)

    if is_simulation:
        settings["poses"]["observation"] = {
            "x":0.17,
            "y":0.,
            "z":0.35,
            "roll":0.0,
            "pitch":1.57,
            "yaw":0.0
        }
    return settings

def robot_learn_poses(robot, names, settings):
    robot.go_to_sleep()
    poses = settings["poses"]
    for name in names:
        print(f"position {name}")
        input(" Press Enter to continue...")

        pose = vars(robot.get_pose())
        poses[name] = pose

    print("everything setup")
    with open(settings["pose_filename"], "w") as f:
        json.dump(poses, f)


def crop_image_border(image, xy_xoffset):
    xoffset, yoffset = xy_xoffset
    return image[yoffset:-yoffset, xoffset:-xoffset]




if __name__ == '__main__':
    main()
