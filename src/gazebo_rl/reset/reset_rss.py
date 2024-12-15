import armpy
import rospy


print(f"Verifying study topics are published...")
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

print(f"\tExpected topics: {REQUIRED_TOPICS}")
print(f"\tExpected cameras: 2")

camera_topics = []
for topic_name, msg_type in rospy.get_published_topics():
    if 'camera_obs__dev_video' in topic_name:
        camera_topics.append(topic_name)

    if topic_name in REQUIRED_TOPICS:
        REQUIRED_TOPICS.remove(topic_name)

assert len(camera_topics) == 2, f"\033[91mERROR: Expected 2 camera topics, found {camera_topics}. Are both cameras running?\033[0m"
assert len(REQUIRED_TOPICS) == 0, f"\033[91mERROR: Missing required topics: {REQUIRED_TOPICS}\033[0m"


print(f"\tFound camera topics: {camera_topics}")

# add colored output
print("\033[92m" + "Check passed! You should be good to go." + "\033[0m")


print(f"Moving arm to backup position and then to target position.")
backup_position = [0.34551798719466237, -0.8454950565561763, 2.169129261535217, -1.232747441193471, 1.4586096006108726, -1.686383909690952] #, 0.5953540153613426]
target_joint_positions = [0.3268500269015339, -1.4471734542578538, 2.3453266624159497, -1.3502152158191212, 2.209384006676201, -1.5125125137062945] #, -0.0877648122691288]

arm = armpy.initialize('gen3_lite')

print(f"Opening gripper ", end='')
arm.open_gripper(); rospy.sleep(1.0) # deal with problems from switching between vel mode and pos mode.
print(f"Done.")

for tjp in [backup_position, target_joint_positions]:
    print(f"Moving to {tjp}", end='...')
    arm.goto_joint_pose(tjp, radians=True, block=False)
    rospy.sleep(5.0)
    print(f"finished waiting. No guarantee that the arm is in position.")
