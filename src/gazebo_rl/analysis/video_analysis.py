import cv2
from pathlib import Path
import numpy as np
from collections import defaultdict
import rosbag
from kortex_driver.msg import BaseCyclic_Feedback
from sensor_msgs.msg import JointState

def video_analysis(base_dir, start_num=None, end_num=None):

    if start_num is None:
        start_num = 0
    if end_num is None:
        end_num = 1e6

    # get every directory that starts with 'user_'
    user_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('user_')]

    # exclude directories whose id is not a number
    user_dirs = [d for d in user_dirs if d.name[5:].isdigit()]
    # exclude directories whose id is not between start_num and end_num
    user_dirs = [d for d in user_dirs if start_num <= int(d.name[5:]) <= end_num]
    # sort them
    user_dirs.sort()

    # read the bag file in each directory and grab the first JointState message
    first_feedbacks = defaultdict(list)
    for user_dir in user_dirs:
        bag_files = list(user_dir.glob("*.bag"))
        if len(bag_files) == 0:
            print(f"Skipping {user_dir.name} because it does not contain any bag files")
            continue

        bag_file = bag_files[0]
        bag = rosbag.Bag(str(bag_file))
        for topic, msg, t in bag.read_messages():
            if topic == "/my_gen3_lite/base_feedback/joint_state":
                first_feedbacks[user_dir.name].append(msg)
                break
        bag.close()

    # average the first feedbacks
    # joint_feedbacks = [np.array(feedbacks.position) for feedbacks in first_feedbacks]
    # import matplotlib.pyplot as plt
    # y = 
    # plt.scatter(range(len(joint_feedbacks)), joint_feedbacks)

    # mean_joint_feedback = np.mean(joint_feedbacks, axis=0)
    # std_joint_feedback = np.std(joint_feedbacks, axis=0)
    # print([f'{entry:+1.2f}' for entry in mean_joint_feedback], [f'{entry:+1.2f}' for entry in std_joint_feedback])

    first_frames = defaultdict(list)
    for user_dir in user_dirs:
        # check whether the directory contains at least two subdirectories taht start with 'cam'
        cam_dirs = [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith('cam')]
        if len(cam_dirs) < 2:
            print(f"Skipping {user_dir.name} because it does not contain at least two cam directories")
            continue

        # grab the first frames from the videos in the subdirectories
        for cam_dir in cam_dirs:
            video_files = list(cam_dir.glob("*.mp4"))
            if len(video_files) == 0:
                print(f"Skipping {cam_dir.name} because it does not contain any video files")
                continue

            video_file = video_files[0]
            cap = cv2.VideoCapture(str(video_file))
            ret, frame = cap.read()
            if ret:
                # print(f"Grabbing first frame from {video_file} {frame.shape}")
                first_frames[user_dir.name].append(frame)
                first_frames[cam_dir.name].append(frame)

                # formatted initial positions
                # initial_positions = [f'{entry:+1.2f}' for entry in first_feedbacks[user_dir.name][0].position]

                # add the formatted first feedback to the frame
                # cv2.putText(frame, f"{initial_positions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # show the frame
                # cv2.imshow(user_dir.name, frame)


                cap.release()
                continue
    # cv2.waitKey(0)
                

    # average the first frames
    for cam_dir_name, frames in first_frames.items():
        print(f"Averaging {len(frames)} frames from {cam_dir_name}")
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        cv2.imshow(cam_dir_name, avg_frame)
    cv2.waitKey(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="~")
    parser.add_argument("--start_num", type=int, default=None)
    parser.add_argument("--end_num", type=int, default=None)
    args = parser.parse_args()

    base_dir = Path("~").expanduser()
    video_analysis(base_dir, args.start_num, args.end_num)