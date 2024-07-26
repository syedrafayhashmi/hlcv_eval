import clip
import torch
from PIL import Image
import numpy as np
import faiss
from sklearn.cluster import KMeans
from pathlib import Path
import os
import shutil
import time
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to get CLIP embeddings for images
def get_clip_embeddings(images, model, preprocess):
    embeddings = []
    for img in images:
        image = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
        embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)

# Function to cluster embeddings
def cluster_embeddings(embeddings, K):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(embeddings)
    return kmeans

# Function to find the nearest cluster center
def find_nearest_cluster_center(embedding, cluster_centers):
    index = faiss.IndexFlatL2(cluster_centers.shape[1])
    index.add(cluster_centers)
    D, I = index.search(embedding, 1)
    return I[0][0]

# Function to match features using LightGlue
def match_features(query_feats, ref_feats, matcher):
    matches = matcher({"image0": query_feats, "image1": ref_feats})
    query_feats, ref_feats, matches = [rbd(x) for x in [query_feats, ref_feats, matches]]
    kpts0, kpts1, matches = query_feats["keypoints"], ref_feats["keypoints"], matches["matches"]
    return len(kpts0[matches[..., 0]])

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Load LightGlue model
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# Set paths
dataset_name = "sku100k_v2"
query_imgs_dir = f"/content/hlcv_eval/lightglue-benchmark_speed/datasets/sku100k_v2/query_images"
class_imgs_dir = f"/content/hlcv_eval/lightglue-benchmark_speed/datasets/sku100k_v2/reference_images"
output_dir_path = f"/content/hlcv_eval/lightglue-benchmark_speed/datasets/sku100k_v2/out/top_1eval4"
# Load reference images and query image
reference_image_paths = [os.path.join(class_imgs_dir, path) for path in os.listdir(class_imgs_dir)]
query_image_paths = [os.path.join(query_imgs_dir, path) for path in os.listdir(query_imgs_dir)]

# Get CLIP embeddings for reference images
reference_images = [Image.open(path) for path in reference_image_paths]
print('generating embeddings')
ref_embeddings = get_clip_embeddings(reference_images, model, preprocess)
print('embeddings done')

# Cluster reference images into K clusters
K = 4  # Number of clusters
kmeans = cluster_embeddings(ref_embeddings, K)
cluster_centers = kmeans.cluster_centers_
ref_labels = kmeans.labels_

# Timing code
start_time = time.time()
n_query_imgs = 0
true_positive = 0

with torch.no_grad():
    print('Starting...')
    
    for query_index, query_img_path in enumerate(query_image_paths):
        print("Query index: ", query_index)
        query_image = Image.open(query_img_path)
        query_embedding = get_clip_embeddings([query_image], model, preprocess)
        
        best_cluster_idx = find_nearest_cluster_center(query_embedding, cluster_centers)
        
        # Extract features for query image
        image0 = load_image(query_img_path)
        query_feats = extractor.extract(image0.to(device))
        
        best_match_score = 0
        best_reference = None
        filenames = {}
        cluster_start_time = time.time()
        
        for idx, label in enumerate(ref_labels):
            if label == best_cluster_idx:
                ref_img_path = reference_image_paths[idx]
                image1 = load_image(ref_img_path)
                ref_feats = extractor.extract(image1.to(device))
                
                match_score = match_features(query_feats, ref_feats, matcher)
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_reference = ref_img_path
                filenames[ref_img_path] = match_score

        cluster_end_time = time.time()
        print(f"Each query image takes: {cluster_end_time - cluster_start_time:.2f} seconds")
        n_query_imgs += 1
        filenames = {k: v for k, v in sorted(filenames.items(), reverse=True, key=lambda item: item[1])}

        threshold = 1  # Top 1 match
        output_dir = os.path.join(output_dir_path, os.path.basename(query_img_path))
        
        if best_match_score >= threshold:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copyfile(query_img_path, os.path.join(output_dir, "queryimg_" + os.path.basename(query_img_path)))
            for index, (reference_img_path, match_score) in enumerate(filenames.items()):
                if index >= threshold:
                    break
                shutil.copyfile(reference_img_path, os.path.join(output_dir, os.path.basename(reference_img_path) + "_" + str(match_score)))
        else:
            print(f"No match found for {query_img_path}")
        
        # Check if the top match is correct
        query_img_basename = os.path.basename(query_img_path)
        if filenames:
            top_match_path = next(iter(filenames.keys()))
            top_match_basename = os.path.basename(top_match_path).split(".")
         
            if query_img_basename.split(".")[1] == top_match_basename[1]:
                true_positive += 1
        
        # Calculate accuracy at runtime
        accuracy = (true_positive / n_query_imgs) * 100
        print(f"Runtime Accuracy after {n_query_imgs} queries: {accuracy:.2f}%")

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")
print("Query images processed: ", n_query_imgs)
print("Total comparisons: ", n_query_imgs * len(ref_labels))
print("Final Top 1 accuracy: ", accuracy)