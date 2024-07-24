import os


dataset_dir = "/Users/naufil/Documents/university/datasets_output/mvimgnet/top_1"
# dataset_dir = "/home/gordian/Desktop/alexander/examples/metatrained_benchmarks/sku100_v2/output_top_5" # DEEPEMD v2
# dataset_dir = "/home/gordian/Desktop/alexander/examples/metatrained_benchmarks/SKU100/output_top_5" # DEEPEMD v1

# dataset_dir = "/home/gordian/Desktop/sku100_v2/output_top_5" # Lightglue v2

total_samples = len(os.listdir(dataset_dir))

true_positive = 0
for folder in os.listdir(dataset_dir):
    if ".DS_Store" not in folder:
        imgs = os.listdir(os.path.join(dataset_dir, folder))
        query_img = f"query_img_{folder}"
        imgs.remove(query_img)
        print(".".join(imgs[0].split("_")[:-2]))
        print(".".join(folder.split("_")[:-1]))
        if ".".join(imgs[0].split("_")[:-2]) == ".".join(folder.split("_")[:-1]):
            true_positive += 1
        break

accuracy = (true_positive/total_samples) * 100
print("Top 1 accuracy: ", accuracy)
