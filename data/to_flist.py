import os
dataset_path = "../datasets/udf/train"
flist = []
for img in os.listdir(dataset_path):
    flist.append(os.path.abspath(os.path.join(dataset_path, img)))
with open("../datasets/udf/train.flist", 'w') as f:
    f.write("\n".join(flist))
print("Done")