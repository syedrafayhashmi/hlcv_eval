import os
import shutil

x = {"49000050141.jpg": 4, "24474381014.jpg": 3, "650348780531.jpg": 2, "74471002108.jpg": 1, "879717001149.jpg": 0, "89836028730.jpg": 0}

reference_imgs_dir = "benchmark/cornerup/reference_images"
output_dir = "output"

for y in os.listdir(reference_imgs_dir):
    for _, (xx, _) in enumerate(x.items()):
        if y == xx:
            print("found it")
            img_path = os.path.join(reference_imgs_dir, y)
            shutil.copyfile(img_path, os.path.join(output_dir, xx))
