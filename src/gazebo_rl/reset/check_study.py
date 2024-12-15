import rospy

REQUIRED_TOPICS = [        
    '/joy',
    # '/my_gen3_lite/in/cartesian_velocity', # NOTE: only published once we start sending commands. If we're not sending this the robot isn't moving and there's obviously a problem
    '/my_gen3_lite/base_feedback',
    '/my_gen3_lite/base_feedback/joint_state',
    '/my_gen3_lite/joint_states',
    '/tf',]

NICE_TO_HAVES = [
    '/reward',
    '/success',
]

print(f"Expected topics: {REQUIRED_TOPICS}")
print(f"Expected cameras: 2")

camera_topics = []
for topic_name, msg_type in rospy.get_published_topics():
    if 'camera_obs__dev_video' in topic_name:
        camera_topics.append(topic_name)

    if topic_name in REQUIRED_TOPICS:
        REQUIRED_TOPICS.remove(topic_name)

assert len(camera_topics) == 2, f"\033[91mERROR: Expected 2 camera topics, found {camera_topics}. Are both cameras running?\033[0m"
assert len(REQUIRED_TOPICS) == 0, f"\033[91mERROR: Missing required topics: {REQUIRED_TOPICS}\033[0m"


print(f"Found camera topics: {camera_topics}")

# add colored output
print("\033[92m" + "Check passed! You should be good to go." + "\033[0m")
