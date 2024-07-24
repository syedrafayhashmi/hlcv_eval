import os

for folder in ["mvimgnet", "mvpn", "sku100k"]:
    folder_path = os.path.join("./", folder)
    top_5_path = os.path.join(folder_path, "top_5")
    top_1_path = os.path.join(folder_path, "top_1")
    top_path = top_5_path # select top 5 or top 1
    for query_folder in os.listdir(top_path):
        print(query_folder)
        if query_folder != ".DS_Store":
            ref_images_path = os.path.join(top_path, query_folder)
            if ".DS_Store" not in ref_images_path:
                for ref_image in os.listdir(ref_images_path):
                    if "query_img" not in ref_image:
                        if folder == "mvpn":
                            os.rename(os.path.join(ref_images_path, ref_image), os.path.join(ref_images_path,ref_image + ".png"))
                        elif folder == "mvimgnet":
                            os.rename(os.path.join(ref_images_path, ref_image), os.path.join(ref_images_path, ref_image + ".jpg"))
                        elif folder == "sku100k":
                            extension = ref_image.split(".")[-1].split("_")[0]
                            os.rename(os.path.join(ref_images_path, ref_image),
                                      os.path.join(ref_images_path, ref_image + f".{extension}"))