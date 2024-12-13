import armpy
import rospy



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
