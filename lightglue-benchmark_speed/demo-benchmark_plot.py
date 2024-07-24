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
query_imgs_dir = "/home/gordian/Desktop/SKU100K/mini_naufil_query_images"
class_imgs_dir = "/home/gordian/Desktop/SKU100K/reference_images_SKU_100K"
output_dir ="/home/gordian/Desktop/SKU100K/output/viz_2048"
# output_dir = "benchmark_results/9 classes"
# query_imgs_dir = "/home/shahram95/Desktop/naufil_ws/SuperGluePretrainedNetwork_experiment/benchmark/product_identification"
# class_imgs_dir = "/home/shahram95/Desktop/naufil_ws/SuperGluePretrainedNetwork_experiment/benchmark/9_classes_concat"
# output_dir = "/home/shahram95/Desktop/naufil_ws/SuperGluePretrainedNetwork_experiment/benchmark_results/9 classes"
# images = Path(query_imgs_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint", **{}).eval().to(device)
# matcher.compile(mode='reduce-overhead')

query_img_path = os.listdir(query_imgs_dir)
class_img_path = os.listdir(class_imgs_dir)

a = time.time()
n_query_imgs = 0

with torch.no_grad():
# with torch.inference_mode():
    for query_img in query_img_path:
        image0 = load_image(os.path.join(query_imgs_dir, query_img))
        feats00 = extractor.extract(image0.to(device))
        os.mkdir(os.path.join(output_dir, query_img))
        n_class_imgs = 0
        max_matching = -1
        filenames = {}
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
            viz_path = os.path.join(os.path.join(output_dir, query_img), class_img)

            axes = viz2d.plot_images([image0, image1])
            viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
            # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
            viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', output_dir, query_img, class_img, len(m_kpts0))

            kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
            viz2d.plot_images([image0, image1])
            viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)



            # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
            # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', viz_path, query_img, class_img, len(m_kpts0))
            #
            # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
            # viz2d.plot_images([image0, image1], query_img)
            # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)