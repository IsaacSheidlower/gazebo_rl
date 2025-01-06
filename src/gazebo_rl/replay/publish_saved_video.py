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
from armpy import kortex_arm
import std_msgs.msg
import armpy

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
    def __init__(self, path, args):
        stop_arm = args.stop_arm
        SHOW_FIRST_FRAME = True

        # dirname is also the name of the camera topic live
        rospy.init_node('bagvideopublisher', anonymous=True)

        if not args.no_arm:
            arm = armpy.initialize('gen3_lite')

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
                print("topic:", topic)
        for topic in video_loader.dirs:
            ros_publisher[topic] = rospy.Publisher(topic, Image, queue_size=1); log_msg_types[topic] = Image
            print(f"Video topic: {topic}")
        for k,v in ros_publisher.items():
            print(k, log_msg_types[k])
        rospy.sleep(0.1)
        ##

        rospy.loginfo(f"Using {get_memory_usage()} MB")

        import time
        
        t0 = None; walltime = rospy.Time.now(); pnum = 0

        if args.no_arm:
            AT_FIRST_POSE = True # don't need to move the arm
        else:
            for topic, msg, t in bag.read_messages('/my_gen3_lite/base_feedback/joint_states'):
                arm.goto_joint_pose(msg.position, radians=True, block=False)
                break
            time.sleep(5)


        crop_dim = rospy.get_param('crop_dim', 0); crop_left_offset = rospy.get_param('crop_left_offset', 0)
        top_index = rospy.get_param('top_index', '4'); bottom_index = rospy.get_param('bottom_index', '0')
        # image_w = None; image_h = None

        for topic, msg, t in bag.read_messages():
            tsec = t.to_sec()
            if t0 is None: # first message 
                t0 = t; walltime = rospy.Time.now(); sleeptime = 0
            elif not args.no_arm: # If we are using the arm, we need to publish the messages in real time
                # how long should we sleep?
                dwalltime = (rospy.Time.now() - walltime).to_sec()
                drostime = (t - t0).to_sec()
                sleeptime = max(0, drostime - dwalltime)
                rospy.sleep(sleeptime)
                walltime = rospy.Time.now(); t0 = t
            else:
                # if we are not using the arm, we can publish as fast as we can
                pass

            if 'cartesian' in topic:
                if stop_arm: continue
            elif 'joy' in topic:
                if args.no_arm:
                    pass
                else:
                    if msg.buttons[4] == 1 or msg.buttons[6] == 1:
                        arm.send_gripper_command(0.1, mode = 'speed', duration = 200, relative=True, block=False)
                    elif msg.buttons[5] == 1 or msg.buttons[7] == 1:
                        arm.send_gripper_command(-10* 0.1, mode = 'speed', duration = 200, relative=True, block=False)
            
            ros_publisher[topic].publish(msg)
            
            camera_frames = video_loader.get_frame_if_available(t)
            for cidx, ctopic in enumerate(video_loader.dirs):
                frame = camera_frames[cidx]
                if frame is not None:
                    if crop_dim > 0:
                        # crop from the right edge for TOP image
                        if top_index in ctopic:
                            frame = frame[:crop_dim, crop_left_offset:crop_dim+crop_left_offset]
                        else:
                            # crop from the left, no offset for BOTTOM image
                            frame = frame[:crop_dim, -crop_dim:]
                            # h, w = frame.shape[:2]
                            # x0 = (w - crop_dim) // 2
                            # y0 = (h - crop_dim) // 2
                            # frame = frame[y0:y0+crop_dim, x0:x0+crop_dim]
                    
                    if args.show_video or SHOW_FIRST_FRAME:
                        cv2.imshow(f'{cidx=} {ctopic=} {frame.shape=}', frame)
                        cv2.waitKey(1)

                    ros_publisher[ctopic].publish(bridge.cv2_to_imgmsg(frame))
                    # cv2.waitKey(1)
                    pnum += 1

            if pnum > 10: SHOW_FIRST_FRAME = False

            rospy.logdebug(f'{tsec:1.2f} {sleeptime:1.3f} {topic}')

        print(f"Published {pnum} frames.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d0', '--dir0', type=str, required=True)
    # parser.add_argument('-d1', '--dir1', type=str, required=True)
    parser.add_argument('-p', '--root', type=str, default='~/')
    parser.add_argument('-dir', '--directory', type=str, required=True)
    parser.add_argument('-s', '--stop-arm', action='store_true')
    parser.add_argument('-na', '--no-arm', action='store_true')
    parser.add_argument('-c', '--crop-dim', type=int, default=0)
    parser.add_argument('-clo', '--crop-left-offset', type=int, default=0)
    parser.add_argument('-v', '--show-video', action='store_true')
    parser.add_argument('-ti', '--top-index', type=str, default='4')
    parser.add_argument('-bi', '--bottom-index', type=str, default='0')
    args = parser.parse_args()

    # print args
    print(f"{args=}")

    rospy.set_param('crop_dim', args.crop_dim)
    rospy.set_param('crop_left_offset', args.crop_left_offset)
    rospy.set_param('top_index', args.top_index)
    rospy.set_param('bottom_index', args.bottom_index)

    path = Path(args.root).expanduser() / args.directory

    print(f"Playing back from {path}")
    # print("hahahah")
    # arm = kortex_arm.Arm()
    # arm.home_arm()
    # print('done')
    import std_msgs
    bag_complete_pub = rospy.Publisher('/playback_complete', std_msgs.msg.Bool, latch=False)
    try:
        BagVideoPublisher(path, args)
    except rospy.ROSInterruptException:
        pass
    finally:
        bag_complete_pub.publish(std_msgs.msg.Bool(True))
        print(f"Stopping")


    # example 