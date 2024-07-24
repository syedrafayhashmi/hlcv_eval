import os
import cv2

videos_dir = "/home/gordian/Desktop/videos"

videos_list = os.listdir(videos_dir)

for x in videos_list:
    video_path = os.path.join(videos_dir, x)
    cap = cv2.VideoCapture(video_path)
    output_dir = os.path.join(os.path.join(videos_dir, "frames"), x)
    os.makedirs(output_dir, exist_ok=True)
    index = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if index % 5 == 0:
            if ret == True:
                cv2.imwrite(os.path.join(output_dir, x[:-4] + "_" + str(index) + ".jpg"), frame)
            else:
                break
        index += 1
    cap.release()
    cv2.destroyAllWindows()
