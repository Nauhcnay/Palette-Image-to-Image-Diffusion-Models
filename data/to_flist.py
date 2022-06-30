import os
import cv2
dataset_path = "../datasets/udf/train"
flist = []
for img in os.listdir(dataset_path):
    p = os.path.join(dataset_path, img)
    a = cv2.imread(p)
    if a.shape[0] < 288 or a.shape[1] < 288:
        continue
    flist.append(os.path.abspath(p))
with open("../datasets/udf/train.flist", 'w') as f:
    f.write("\n".join(flist))
print("Done")