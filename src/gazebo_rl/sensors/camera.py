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
# # sys.path.append(os.path.expanduser('~/workspace/fastrl/nov20/second_wind/'))
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.datasets.image_writer import AsyncImageWriter
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


class Camera():
    output_directory = None
    def __init__(self, camera):
        self.camera = camera
        self.image_writer = AsyncImageWriter(num_processes=0, num_threads=1)

    def output_directory_cb(self, msg):
        if msg.data:
            self.output_directory = Path(msg.data) / f"cam{str(self.camera.port).replace('/', '_')}"
            print(f"{self.output_directory=}")
            if not os.path.exists(str(self.output_directory)):
                print('\tcreating directory.')
                os.mkdir(str(self.output_directory))
        else:
            print(f"Output directory empty.")
            self.output_directory = None

    def run(self):
        self.camera.connect()
        # img_pubs = [rospy.Publisher(f"camera_obs_{str(c.port).replace('/', '_')}", Image, queue_size=1) for c in self.cameras]
        # image_transport_pubs = [image_transport.Publisher(f"camera_obs_{str(c.port).replace('/', '_')}") for c in cameras]
        img_pub = rospy.Publisher(f"camera_obs_{str(self.camera.port).replace('/', '_')}", Image, queue_size=1)
        rospy.Subscriber('/uid_data_dir', String, self.output_directory_cb, queue_size=1)


        bridge = CvBridge()
        RATE_HZ = rospy.get_param("/observation_rate", 30)

        try:
            rospy.init_node("observation_pub", anonymous=True)
        except Exception as e:
            print(e)

        rate = rospy.Rate(RATE_HZ, reset=True)

        hearbeat_period = RATE_HZ * 10; heartbeat = 0
        while not rospy.is_shutdown():
            heartbeat += 1
            if heartbeat % hearbeat_period == 0: print("Observation publisher heartbeat.")
            # print("Observation publisher heartbeat.")


            # if rospy.get_param("/pause", False):
            #     rate.sleep()
            #     print("PAUSED BY PARAMETER. Resetting weight tracking.")
            #     tracked_weights = []; short_term_weights = []
            #     continue

            try:
                # for pub, cam in zip(img_pubs, cameras):
                img = self.camera.async_read()
                # cv2.imshow(f'{str(self.camera.port)} {img.shape}', img); cv2.waitKey(1)
                # resized_image = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
                # img_msg = bridge.cv2_to_imgmsg(resized_image, encoding='rgb8')
                # img_pub.publish(img_msg)

                # write out the images if we have an output_directory
                if self.output_directory and self.image_writer:
                    fn = str(rospy.Time.now())+'.png'
                    fpath = self.output_directory / fn
                    print(f"Writing {fpath}")
                    self.image_writer.save_image(image=img, fpath=fpath)

            except Exception as e:
                print("Error publishing observation.", e)        
            rate.sleep()

    def stop(self):
        self.camera.disconnect()
        if self.image_writer is not None:
            self.image_writer.wait_until_done()
            self.image_writer.stop()
            self.image_writer = None

if __name__ == '__main__':
    import cv2
    sim = False #rospy.get_param("/sim", False)

    # Allow for command line argument
    import sys
    camid = int(sys.argv[1]) if len(sys.argv) == 2 else 0
    camera = OpenCVCamera(camid)

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
        if sim:
            print("Relying on gazebo simulated camera.")
        else:
            savepath = None # "~/tmp/rl_camera_images"
        print("publishing observations")
        c = Camera(camera)
        c.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        print(f"Stopping")
        c.stop()
