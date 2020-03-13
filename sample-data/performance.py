import struct

import imgaug.augmenters as iaa
import imgaug as ia

from tqdm import tqdm
from lib.python.pose_format import PoseReader, Pose
from lib.python.pose_format.custom_augment.piecewise_affine_kp import PiecewiseAffineKP

iterations = 20

buffer = open("1.pose", "rb").read()
p = PoseReader(buffer).read()

print("Unpack file")
for _ in tqdm(range(iterations), total=iterations):
    p = PoseReader(buffer).read()

print("Focus Pose")
for _ in tqdm(range(iterations), total=iterations):
    p.focus_pose()

print("Normalize")
for _ in tqdm(range(iterations), total=iterations):
    p.normalize(
        dist_p1=("pose_keypoints_2d", 2),  # RShoulder
        dist_p2=("pose_keypoints_2d", 5),  # LShoulder
    )

print("Vectorize")
for _ in tqdm(range(iterations), total=iterations):
    list(p.to_vectors(["angle", "distance"]))

print("Augment")
for _ in tqdm(range(iterations), total=iterations):
    seq = iaa.Sequential([])
    p.augment2d(seq)
