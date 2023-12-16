from pyniryo import *
import json
import segment

def main():

    # initialize robot
    settings = get_settings(is_simulation=True)
    robot = NiryoRobot(settings["ip"])
    robot.calibrate_auto()
    robot.update_tool()
    robot.release_with_tool()
    conveyor_id = robot.set_conveyor()

    # define which segmentation
    # method to use
    infer = segment.color_seg

    # sorting loop
    while True:

        # move to observation pose
        observation_pose = settings["poses"]["observation"]
        robot.move_pose(PoseObject(**observation_pose))

        # preprocess incoming image
        xy_workspace_offset = (15, 15)
        mtx, dist        = robot.get_camera_intrinsics()
        img_compressed   = robot.get_img_compressed()
        img_raw          = uncompress_image(img_compressed)
        img_undistorted  = undistort_image(img_raw, mtx, dist)
        img_workspace    = extract_img_workspace(img_undistorted, workspace_ratio=1.0)
        image            = crop_image_border(img_workspace, xy_workspace_offset)

        # infer which objects are present
        inference_result = infer(image)
        names    = inference_result["names"]
        contours = inference_result["contours"]

        # if no objects are present
        # the conveyor belt can move
        # a little bit and we start all over
        if len(contours) == 0:
            print("no contours")
            robot.run_conveyor(conveyor_id)
            robot.wait(0.5)
            robot.stop_conveyor(conveyor_id)
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
        xoffset, yoffset = xy_workspace_offset
        cx += xoffset
        cy += yoffset
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
        settings["workspace_name"] = "t_workspace"

    settings["pose_filename"] = "./resources/poses.json"

    with open(settings["pose_filename"], "r") as f:
        settings["poses"] = json.load(f)

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





































