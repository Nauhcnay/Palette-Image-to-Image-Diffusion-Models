import os
import cv2
import random

dataset_path = "../datasets/udf/train"
flist = []
for img in os.listdir(dataset_path):
    p = os.path.join(dataset_path, img)
    a = cv2.imread(p)
    if a.shape[0] < 288 or a.shape[1] < 288:
        continue
    flist.append(os.path.abspath(p))

random.shuffle(flist)
train_flist = flist[:-32]
test_flist = flist[-32:]
with open("../datasets/udf/train.flist", 'w') as f:
    f.write("\n".join(train_flist))
with open("../datasets/udf/test.flist", 'w') as f:
    f.write("\n".join(test_flist))
print("Done")