#!/home/j/workspace/lerobot/venv/bin/python


import rospy
import rosbag
import os, sys, resource
from pathlib import Path
import time

import cv2
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image, Joy
from collections import deque

import std_msgs.msg

def get_memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == "posix":
        # On Linux and macOS, ru_maxrss is in kilobytes
        usage = usage / 1024  # Convert to MB
    return usage

class VideoLoader:
    def __init__(self, video_dirs, threshold_ns=0., cache_size=100):
        """
        Initialize the VideoLoader with paths to videos and their corresponding timestamps.
        """
        self.dirs = [str(entry).split('/')[-1] for entry in video_dirs]
        self.video_paths = [Path(dirname).absolute() / 'output.mp4' for dirname in video_dirs]
        timestamp_paths = [Path(dirname).absolute() / 'video_frame_timestamps.txt' for dirname in video_dirs]
        self.timestamp_lists = []
       
        for timestamp_fn in timestamp_paths:
            timestamps = []
            with open(str(timestamp_fn), 'r') as fp:
                timestamps = fp.readlines()
            timestamps = [rospy.Time.from_seconds(int(t) / 1e9) for t in timestamps]
            self.timestamp_lists.append(timestamps)

        self.frames = [deque(maxlen=100) for _ in self.video_paths]  # To store frames of each video
        self.captures = []
        self.frame_caches = [deque(maxlen=cache_size) for _ in self.video_paths]
        # self.frame_timestamps = []  # To store timestamp lists of each video
        self.frame_idx = [0 for _ in self.video_paths] # NOTE: this class dumps its frames, it doesn't search them, when it runs into problems it yells and skips frames. Its like a bag.
        # self._load_videos()
        self._open_videos()



        self.threshold_ns = rospy.Time(secs=0, nsecs=threshold_ns)

    def _load_videos(self):
        """
        Load the videos into memory and store their frames and timestamps. Too big to use with big videos :(
        """
        for video_path, timestamps in zip(self.video_paths, self.timestamp_lists):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            video_frames = []
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_index < len(timestamps):
                    video_frames.append((timestamps[frame_index], frame))
                frame_index += 1

            cap.release()
            self.frames.append(video_frames)
            self.frame_timestamps.append(timestamps)

    def _open_videos(self):
        """
        Open video files for streaming.
        """
        self.captures = [cv2.VideoCapture(str(path)) for path in self.video_paths]
        for i, cap in enumerate(self.captures):
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {self.video_paths[i]}")
            else:
                # load the first frame
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Cannot read from capture {self.video_paths[i]}")
                self.frames[i].append(frame)


    # NOTE: Better to drop frames than spend too much time getting them (frames will be dropped in the real world)
    def get_frame_if_available(self, target_timestamp):
        """
        Retrieve a frame if the next frame is close enough to the passed target_timestamp. NOTE: if you didn't have a fast signal in the rosbag you would miss 
        
        :param target_timestamp: Timestamp for which to retrieve the frame.
        :return: Frames at the given target_timestamp or None if no close match.
        """
        t0 = time.perf_counter()
        ret_frames = [None for _ in self.frames]
        # for cam_idx, cam_frames in enumerate(self.frames):
        #     idx = self.frame_idx[cam_idx]
        #     if idx >= len(cam_frames): 
        #         continue
        #     ts = self.frame_timestamps[cam_idx][idx]
        read = 0
        for cam_idx, capture in enumerate(self.captures):
            idx = self.frame_idx[cam_idx]
            if len(self.frames[cam_idx]) == 0:
                ret, frame = capture.read(); read += 1
                self.frames[cam_idx].append(frame)

            if idx > len(self.timestamp_lists[cam_idx]):
                rospy.loginfo("Out of video frames for camera {self.video_paths[cam_idx]}")
                continue

            ts = self.timestamp_lists[cam_idx][idx]
            if (ts - self.threshold_ns).to_nsec() <= target_timestamp.to_nsec(): # We're late, on time, or ahead <= threshold_ns
                # ret_frames[cam_idx] = cam_frames[idx][1]
                ret_frames[cam_idx] = self.frames[cam_idx].pop()
                self.frame_idx[cam_idx] = self.frame_idx[cam_idx] + 1

        endT = time.perf_counter()
        if (endT - t0) > 0.01:
            rospy.logwarn(f"Video read took longer than 0.01 seconds: {(endT - t0)=}")
        return ret_frames
    

class BagVideoPublisher():
    def __init__(self, path):
        # dirname is also the name of the camera topic live
        rospy.init_node('bagvideopublisher', anonymous=True)


        bridge = cv_bridge.CvBridge()
        bag = rosbag.Bag(str(path / 'trial_data.bag'))

        print(f"Loading videos...", end='')
        video_dirs = [entry for entry in path.iterdir() if "cam_dev_video" in str(entry)]
        print(f"{video_dirs}", end=' -- ')
        video_loader = VideoLoader(video_dirs) # TODO: Need to buffer and stream as each video is about 8GB of RAM when preloaded and held as frames
        print(f"Loaded.")

        # Set-up ros publishers
        ros_publisher = {}; log_msg_types = {}
        for topic, msg, t in bag.read_messages():
            if topic in ros_publisher: continue
            else:
                ros_publisher[topic] = rospy.Publisher(topic, type(msg), queue_size=1); log_msg_types[topic] = type(msg)
        for topic in video_loader.dirs:
            ros_publisher[topic] = rospy.Publisher(topic, Image, queue_size=1); log_msg_types[topic] = Image

        for k,v in ros_publisher.items():
            print(k, log_msg_types[k])
        rospy.sleep(0.1)
        ##

        rospy.loginfo(f"Using {get_memory_usage()} MB")

        import time
        t0 = None; walltime = rospy.Time.now(); pnum = 0
        for topic, msg, t in bag.read_messages():
            tsec = t.to_sec()
            if t0 is None: # first message 
                t0 = t; walltime = rospy.Time.now(); sleeptime = 0
            else:
                # how long should we sleep?
                dwalltime = (rospy.Time.now() - walltime).to_sec()
                drostime = (t - t0).to_sec()
                sleeptime = max(0, drostime - dwalltime)
                rospy.sleep(sleeptime)
                walltime = rospy.Time.now(); t0 = t


            ros_publisher[topic].publish(msg)

            camera_frames = video_loader.get_frame_if_available(t)
            for cidx, ctopic in enumerate(video_loader.dirs):
                frame = camera_frames[cidx]
                if frame is not None:
                    aframe = cv2.putText(frame, f'{tsec:1.5f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1., (255,0,0), 2)
                    cv2.imshow(f'{cidx=}', aframe)
                    ros_publisher[ctopic].publish(bridge.cv2_to_imgmsg(frame))
                    cv2.waitKey(1)
                    pnum += 1

            rospy.logdebug(f'{tsec:1.2f} {sleeptime:1.3f} {topic}')

        print(f"Published {pnum} frames.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d0', '--dir0', type=str, required=True)
    # parser.add_argument('-d1', '--dir1', type=str, required=True)
    parser.add_argument('-p', '--root', type=str, default='~/')
    parser.add_argument('-dir', '--directory', type=str, required=True)
    args = parser.parse_args()

    path = Path(args.root).expanduser() / args.directory

    print(f"Playing back from {path}")

    import std_msgs
    bag_complete_pub = rospy.Publisher('/playback_complete', std_msgs.msg.Bool, latch=False)
    try:
        BagVideoPublisher(path)
    except rospy.ROSInterruptException:
        pass
    finally:
        bag_complete_pub.publish(std_msgs.msg.Bool(True))
        print(f"Stopping")