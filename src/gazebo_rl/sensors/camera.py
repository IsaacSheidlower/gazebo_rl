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
# # from mbrl.camera import Camera, CAMERA_CFG, SIDE_CAMERA_CFG, detect_circles
from cv_bridge import CvBridge
# from std_msgs.msg import Int32MultiArray, Float32MultiArray, Bool
# import collections
# import cv2
# import threading
# import tf2_ros

import datetime, time
# import os
# # sys.path.append(os.path.expanduser('~/workspace/fastrl/nov20/second_wind/'))
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera


def observation_publisher(cameras, show=False, picture_in_picture=False):
    img_pubs = [rospy.Publisher(f"camera_obs_{str(c.port).replace('/', '_')}", Image, queue_size=1) for c in cameras]

    bridge = CvBridge()
    RATE_HZ = rospy.get_param("/observation_rate", 100)

    try:
        rospy.init_node("observation_pub", anonymous=True)
    except:
        pass
    rate = rospy.Rate(RATE_HZ, reset=True)

    hearbeat_period = RATE_HZ * 120; heartbeat = 0
    while not rospy.is_shutdown():
        heartbeat += 1
        if heartbeat % hearbeat_period == 0: print("Observation publisher heartbeat.")

        # if rospy.get_param("/pause", False):
        #     rate.sleep()
        #     print("PAUSED BY PARAMETER. Resetting weight tracking.")
        #     tracked_weights = []; short_term_weights = []
        #     continue

        try:
            for pub, cam in zip(img_pubs, cameras):
                img = cam.async_read()
                cv2.imshow(str(cam.port), img); cv2.waitKey(1)
                img_msg = bridge.cv2_to_imgmsg(img, encoding='rgb8')
                pub.publish(img_msg)
        except Exception as e:
            print("Error publishing observation.", e)        
        rate.sleep()

if __name__ == '__main__':
    import cv2
    sim = False #rospy.get_param("/sim", False)
    camera = OpenCVCamera(1)

    cameras = [camera]
    [camera.connect() for c in cameras]

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
        observation_publisher(cameras)
    except rospy.ROSInterruptException:
        pass
    finally:
        camera.disconnect()