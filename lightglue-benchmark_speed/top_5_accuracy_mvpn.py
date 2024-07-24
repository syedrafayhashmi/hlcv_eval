import os

dataset_dir = "./datasets/output/mvpn/top_5"

total_samples = len(os.listdir(dataset_dir))

true_positive = 0
for folder in os.listdir(dataset_dir):
    imgs = os.listdir(os.path.join(dataset_dir, folder))
    query_img = f"query_img_{folder}"
    imgs.remove(query_img)
    # print("_".join(imgs[0].split("_")[:2]))
    # print(folder.split(".png")[0])
    # print("_".join(folder.split("_")[:2]))
    found = False
    for img in imgs:
        if "_".join(img.split("_")[:2]) == "_".join(folder.split("_")[:2]):
            true_positive += 1
            found = True
            break
    if not found:
        print(folder)

accuracy = (true_positive/total_samples) * 100
print("Top 5 accuracy: ", accuracy)

# DeepEMD metatrained
# Top 1 accuracy:  40.909090909090914
# Top 5 accuracy:  68.18181818181817

# DeepEMD pretrained
# Top 1 accuracy:  38.63636363636363
# Top 5 accuracy:  68.18181818181817

# Lightglue
# Top 1 accuracy:  59.09090909090909
# Top 5 accuracy:  77.27272727272727

