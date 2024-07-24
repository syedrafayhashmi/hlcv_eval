# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path

# if Path.cwd().name != "LightGlue":
#     !git clone --quiet https://github.com/cvg/LightGlue/
#     %cd LightGlue
#     !pip install --progress-bar off --quiet -e .

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import os
import time
import shutil

torch.set_grad_enabled(False)
dataset_name = "mvimgnet"
query_imgs_dir = f"./datasets/{dataset_name}/query_images"
class_imgs_dir = f"./datasets/{dataset_name}/reference_images"
threshold = 1
output_dir_path = f"./datasets/output/{dataset_name}/top_{threshold}"
max_keypoints = 1024
# query_imgs_dir = "/home/gordian/Desktop/sku100_v2/query_images"
# class_imgs_dir = "/home/gordian/Desktop/sku100_v2/reference_images"
# output_dir = "benchmark_results/9 classes"
# query_imgs_dir = "/home/shahram95/Desktop/naufil_ws/SuperGluePretrainedNetwork_experiment/benchmark/product_identification"
# class_imgs_dir = "/home/shahram95/Desktop/naufil_ws/SuperGluePretrainedNetwork_experiment/benchmark/9_classes_concat"
# output_dir = "/home/shahram95/Desktop/naufil_ws/SuperGluePretrainedNetwork_experiment/benchmark_results/9 classes"
# images = Path(query_imgs_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint", **{}).eval().to(device)
# matcher.compile(mode='reduce-overhead')

query_img_path = os.listdir(query_imgs_dir)
class_img_path = os.listdir(class_imgs_dir)

a = time.time()
n_query_imgs = 0

with torch.no_grad():
# with torch.inference_mode():
    for query_index, query_img in enumerate(query_img_path):
        print("Query index: ", query_index)
        image0 = load_image(os.path.join(query_imgs_dir, query_img))
        feats00 = extractor.extract(image0.to(device))
        # os.mkdir(os.path.join(output_dir, query_img))
        n_class_imgs = 0
        max_matching = -1
        filenames = {}
        y = time.time()
        for class_img in class_img_path:
            try:
                image1 = load_image(os.path.join(class_imgs_dir, class_img))
            except:
                continue
            feats1 = extractor.extract(image1.to(device))
            matches01 = matcher({"image0": feats00, "image1": feats1})
            feats0, feats1, matches01 = [
                rbd(x) for x in [feats00, feats1, matches01]
            ]  # remove batch dimension

            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
            max_matching = max(max_matching, len(m_kpts0))
            filenames[class_img] = len(m_kpts0)
            n_class_imgs += 1
            # print(n_class_imgs)
        z = time.time()
        print("Each query image takes: ", z - y)
        n_query_imgs += 1
        filenames = {k: v for k, v in sorted(filenames.items(), reverse=True, key=lambda item: item[1])}
        # print("matched filenames")
        for index, (reference_img_name, y) in enumerate(filenames.items()):
            if index >= threshold:
                break
            # print(reference_img_name, y)
            # print(class_img_path)
            # print("reference image name: ", reference_img_name)
            ref_img_path = os.path.join(class_imgs_dir, reference_img_name)
            que_img_path = os.path.join(query_imgs_dir, query_img)
            output_dir = os.path.join(output_dir_path, query_img)
            print(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            shutil.copyfile(que_img_path, os.path.join(output_dir, "query_img_" + query_img))
            shutil.copyfile(ref_img_path, os.path.join(output_dir, reference_img_name + "_" + str(y)))

b = time.time()
print("Total seconds: ", b - a)
print("Query images: ", n_query_imgs)
print("Class images: ", n_class_imgs)
print("total comparisons: ", n_query_imgs * n_class_imgs)
        # axes = viz2d.plot_images([image0, image1])
        # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', output_dir, query_img, class_img, len(m_kpts0))

        # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        # viz2d.plot_images([image0, image1], query_img)
        # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
