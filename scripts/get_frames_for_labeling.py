import cv2
import pandas as pd
import numpy as np
import os


df_bad = pd.read_csv('../dataframes/false_class_df.csv')  # датасет с неверно классифицированными видео
df_good = pd.read_csv('../dataframes/true_class_df.csv') # датасет с верно классифицированными видео
path_to_images = '../images_for_labeling'


def save_frames(video_path):
    """
    Функция для получения фреймов из видео для дальнейшей разметки
    :param video_path: str
    :return: None
    """
    path, file = video_path.rsplit('\\', 1)
    name_file = file.split('.')[0]
    class_name = path.rsplit('/', 1)[1]

    cap = cv2.VideoCapture(video_path)
    count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Если класс видео ('bridge_down', 'bridge_up', 'no_action') => берем только один центральный кадр
    if class_name in ('bridge_down', 'bridge_up', 'no_action'):
        n_frame = count_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
        ret, frame = cap.read()

        # выполняем повышение контрастности для удаления засвеченности
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)  # apply CLAHE to the L-channel
        lab = cv2.merge((l2, a, b))  # merge channels
        frame_new = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.join(path_to_images, f'{name_file}_{n_frame}.jpg'), frame_new)
    else:
        # Если класс видео 'train_in_out' => берем 5 равномерно распределенных кадров
        frames = np.linspace(1, count_frames, 5, dtype='int32')
        for n_frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
            ret, frame = cap.read()

            # выполняем повышение контрастности для удаления засвеченности
            clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)  # apply CLAHE to the L-channel
            lab = cv2.merge((l2, a, b))  # merge channels
            frame_new = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            cv2.imwrite(os.path.join(path_to_images, f'{name_file}_{n_frame}.jpg'), frame_new)


for video in df_bad['path']:
    save_frames(video)

for video in df_good['path']:
    save_frames(video)
