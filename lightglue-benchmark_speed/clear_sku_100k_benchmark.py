import os
import shutil

output_dir = "/home/gordian/Desktop/SKU100K/output_top_5"
reference_dir = "/home/gordian/Desktop/SKU100K/reference_images"
dump_dir = "/home/gordian/Desktop/SKU100K/output_top_55"

# total_samples = len(os.listdir(dataset_dir))
count = 0
reference_imgs = os.listdir(reference_dir)
reference_imgs = [".".join(x.split(".")[:-1]) for x in reference_imgs]

for img in os.listdir(output_dir):
    if ".".join(img.split(".")[:-1]) in reference_imgs:
        count += 1
        shutil.copytree(os.path.join(output_dir, img), os.path.join(dump_dir, img))

print(count)