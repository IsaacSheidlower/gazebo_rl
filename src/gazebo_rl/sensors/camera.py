#!/home/j/workspace/lerobot/venv/bin/python

# import sys
# print(sys.executable)
# print(sys.version)

import rospy

# import numpy as np
# import rospy
# from gazebo_rl.msg import ObsMessage

# from kortex_driver.srv import *
# from kortex_driver.msg import *
# from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String
# # from mbrl.camera import Camera, CAMERA_CFG, SIDE_CAMERA_CFG, detect_circles
from cv_bridge import CvBridge
import image_transport
# from std_msgs.msg import Int32MultiArray, Float32MultiArray, Bool
# import collections
# import cv2
# import threading
# import tf2_ros

import datetime, time
import os
from pathlib import Path
import numpy as np

from videorecorder import VideoRecorder
# # sys.path.append(os.path.expanduser('~/workspace/fastrl/nov20/second_wind/'))
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
# from lerobot.common.datasets.image_writer import AsyncImageWriter

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

class Camera():
    output_directory = None
    def __init__(self, camera):
        self.camera = camera
        # self.image_writer = AsyncImageWriter(num_processes=0, num_threads=1)
        self.video_recorder = None
        self.timestamp_fp = None
        self.frame_times = [] # for debugging

    def output_directory_cb(self, msg):
        if msg.data:
            self.output_directory = Path(msg.data) / f"cam{str(self.camera.port).replace('/', '_')}"
            print(f"{self.output_directory=}")
            if not os.path.exists(str(self.output_directory)):
                print('\tcreating directory.')
                os.mkdir(str(self.output_directory))
        else:
            print(f"Empty output directory message. Stopping.")
            self.output_directory = None
            self.stop()
            # if self.video_recorder:
            #     self.video_recorder.stop()
            #     print(f"{self.video_recorder.is_alive()=}")
            #     self.video_recorder = None
            # if self.timestamp_fp: self.timestamp_fp.close()


    def run(self):
        rospy.loginfo_once("Camera Running")
        self.camera.connect()

        # img_pubs = [rospy.Publisher(f"camera_obs_{str(c.port).replace('/', '_')}", Image, queue_size=1) for c in self.cameras]
        # image_transport_pubs = [image_transport.Publisher(f"camera_obs_{str(c.port).replace('/', '_')}") for c in cameras]
        pub_topic = f"camera_obs_{str(self.camera.port).replace('/', '_')}_96x96"
        img_pub = rospy.Publisher(pub_topic, Image, queue_size=1)
        print(f"Publishing on {pub_topic}")

        rospy.Subscriber('/uid_data_dir', String, self.output_directory_cb, queue_size=1)
        bridge = CvBridge()
        try:
            rospy.init_node("observation_pub", anonymous=True)
        except Exception as e:
            print(e)

        RATE_HZ = rospy.get_param("/observation_rate", 30)
        hearbeat_period = RATE_HZ * 10; heartbeat = 0
        while not rospy.is_shutdown():
            heartbeat += 1
            if heartbeat % hearbeat_period == 0: print(f"heartbeat {time.time():1.2f}")
            # print("Observation publisher heartbeat.")

            try:
                # for pub, cam in zip(img_pubs, cameras):
                img = self.camera.read()
                frame_time = rospy.Time.now()

                resized_image = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
                
                # ret, img = self.cap.read()
                # if not ret:
                #     print("Failed to read frame from the camera.")
                #     break

                cv2.imshow(f'{str(self.camera.port)} {img.shape}', img); cv2.waitKey(10)
                img_msg = bridge.cv2_to_imgmsg(resized_image, encoding='bgr8')
                img_pub.publish(img_msg)

                # write out the images if we have an output_directory
                if self.output_directory:
                    # if self.image_writer:
                    #     fn = str(rospy.Time.now())+'.png'
                    #     fpath = self.output_directory / fn
                    #     print(f"Writing {fpath}")
                    #     self.image_writer.save_image(image=img, fpath=fpath)
                    if self.video_recorder:
                        self.video_recorder.write_frame(img)
                        self.timestamp_fp.write(str(frame_time)+'\n')
                        self.frame_times.append(frame_time.to_sec())
                    else:
                        print(f"Creating video recorder.")
                        self.video_recorder = VideoRecorder(
                            video_file=rospy.get_param('~video_file', str(self.output_directory) + '/output.mp4'),
                            codec=rospy.get_param('~codec', 'mp4v'),
                            fps=rospy.get_param('~fps', self.camera.fps),
                            frame_size=(
                                rospy.get_param('~frame_width', self.camera.width),
                                rospy.get_param('~frame_height', self.camera.height)
                            ),
                            max_queue_size=rospy.get_param('~max_queue_size', 100)
                        )
                        self.video_recorder.start()
                        self.timestamp_fp = open(str(self.output_directory / 'video_frame_timestamps.txt'), 'w')

            except Exception as e:
                print("Error publishing observation.", e)        

    def stop(self):
        # self.camera.disconnect()
        # if self.image_writer is not None:
        #     self.image_writer.wait_until_done()
        #     self.image_writer.stop()
        #     self.image_writer = None
        if self.video_recorder: 
            self.video_recorder.stop(); self.video_recorder = None
            if len(self.frame_times) >= 2:
                print(f"Recorded from {self.frame_times[0]} to {self.frame_times[-1]}. Duration {self.frame_times[-1] - self.frame_times[0]:1.2f}sec")
            self.frame_times = [] # for debugging
        if self.timestamp_fp: 
            self.timestamp_fp.close(); self.timestamp_fp = None

        

if __name__ == '__main__':
    import cv2
    cv2.destroyAllWindows()
    sim = False #rospy.get_param("/sim", False)

    # Allow for command line argument
    import sys
    camid = int(sys.argv[1]) if len(sys.argv) == 2 else 0

    # camid = rospy.get_param('~camid', 0)

    print(f"{camid=}")
    # camid = f'/dev/camera{sys.argv[1]}' if len(sys.argv) == 2 else 0
    cfg = OpenCVCameraConfig(CAMERA_FPS, CAMERA_WIDTH, CAMERA_HEIGHT)
    camera = OpenCVCamera(camid, cfg)

    # repo_id = 'aabl_test'
    # LeRobotDatasetMetadata.create(repo_id, fps=30)
    # dataset = LeRobotDataset(repo_id=repo_id, local_files_only=True)
    # print(f'{dataset.features=}')

    cameras = [camera]

    # picture_in_picture = rospy.get_param("/picture_in_picture", False)
    # print("sim: ", sim, end=' ')
    # camera, side_camera = None, None
    # for _ in range(1000):
    #     img = camera.async_read()
    #     cv2.imshow('img', img)
    #     cv2.waitKey(1)

    try:
        print("publishing observations")
        c = Camera(camera)
        c.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        print(f"Stopping")
        c.camera.disconnect()
