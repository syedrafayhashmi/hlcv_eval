import os

dataset_dir = "./datasets/output/mvpn/top_1"

total_samples = len(os.listdir(dataset_dir))

true_positive = 0
for folder in os.listdir(dataset_dir):
    imgs = os.listdir(os.path.join(dataset_dir, folder))
    query_img = f"query_img_{folder}"
    imgs.remove(query_img)
    print("_".join(imgs[0].split("_")[:2]))
    # print(folder.split(".png")[0])
    print("_".join(folder.split("_")[:2]))
    if "_".join(imgs[0].split("_")[:2]) == "_".join(folder.split("_")[:2]):
        true_positive += 1

accuracy = (true_positive/total_samples) * 100
print("Top 1 accuracy: ", accuracy)


# DEEPEMD metatrained
# Top 1 accuracy:  40.909090909090914


#DEEPEMD pretrained
# Top 1 accuracy:  38.63636363636363


# Lightglue
# Top 1 accuracy:  59.09090909090909
