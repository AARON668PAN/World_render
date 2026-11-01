"""Unitree G1 + Dex3 LeRobot observation.state names and the matching G1 URDF joint names."""

STATE_TO_JOINT = {
    **{f"k{S}{J}": f"{s}_{j}_joint" for S, s in (("Left", "left"), ("Right", "right"))
       for J, j in (("ShoulderPitch", "shoulder_pitch"), ("ShoulderRoll", "shoulder_roll"), ("ShoulderYaw", "shoulder_yaw"),
                    ("Elbow", "elbow"), ("WristRoll", "wrist_roll"), ("WristPitch", "wrist_pitch"), ("WristYaw", "wrist_yaw"))},
    **{f"k{S}Hand{F}{i}": f"{s}_hand_{f}_{i}_joint" for S, s in (("Left", "left"), ("Right", "right"))
       for F, f in (("Thumb", "thumb"), ("Index", "index"), ("Middle", "middle")) for i in range(3)},
}
STATE_NAMES = ["kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow", "kLeftWristRoll",
               "kLeftWristPitch", "kLeftWristYaw", "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
               "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw", "kLeftHandThumb0",
               "kLeftHandThumb1", "kLeftHandThumb2", "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0",
               "kLeftHandIndex1", "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2", "kRightHandIndex0",
               "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1"]
