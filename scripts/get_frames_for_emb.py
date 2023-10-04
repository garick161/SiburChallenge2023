from get_problem_frames import contrast_increase
import cv2
import pandas as pd
import numpy as np
import os


def get_frames_and_emb(video: str):
    """
    Функция берет 1 кадр в секунду, производит детекцию объектов, наносит bounding_boxes
    """
    file_name = video.split('.')[0]
    cap = cv2.VideoCapture(f'{path_to_video}/{name_class}/{video}')
    count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frame = count_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)

    while True:
        ret, frame = cap.read()
        if ret:
            frame_new = contrast_increase(frame)
            cv2.imwrite(os.path.join(path_to_img_for_emb, f'{name_class}/{file_name}.jpg'), frame_new)
        else:  # кадры закончились
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == '__main__':
    path_to_video = '../prepair_dataset/train'
    path_to_img_for_emb = '../images_for_emb'

    name_class = 'no_action'
    video_list = [f for f in os.listdir(os.curdir) if os.path.isfile(f) and f.endswith('.mp4')]

    for video in video_list:
        get_frames_and_emb(video)

