import os
import shutil

folder_path = os.path.join("./", "mvimgnet")
top_5_path = os.path.join(folder_path, "top_5")
top_1_path = os.path.join(folder_path, "top_1")
# top_path = top_5_path # select top 5 or top 1
for query_folder in os.listdir(top_5_path):
    print(query_folder)
    if query_folder != ".DS_Store":
        ref_images_path = os.path.join(top_5_path, query_folder)
        if ".DS_Store" not in ref_images_path:
            ref_images = os.listdir(ref_images_path)
            max_matched_keypoints = -1
            matched_images = {}
            for ref_image in ref_images:
                matched_keypoints = ref_image.split("_")[-1].split(".")[0]
                matched_images[ref_image] = matched_keypoints
            sorted_dict_desc = {k: v for k, v in sorted(matched_images.items(), key=lambda item: item[1], reverse=True)}
            top_matched_image = next(iter(sorted_dict_desc.items()))
            top_1_ref_images_path = os.path.join(top_1_path, query_folder)
            os.makedirs(top_1_ref_images_path, exist_ok=True)
            shutil.copy(os.path.join(ref_images_path, top_matched_image[0]), os.path.join(top_1_ref_images_path, top_matched_image[0]))
            shutil.copy(os.path.join(ref_images_path, "query_img_" + query_folder),
                        os.path.join(top_1_ref_images_path, "query_img_" + query_folder))