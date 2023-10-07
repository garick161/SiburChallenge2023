import pandas as pd
import numpy as np
import cv2
import os
from contrast_increase import contrast_increase


def frames_for_labeling(path_to_df: str):
    """Функция для получения фреймов из видео для дальнейшей разметки"""
    df = pd.read_csv(path_to_df)

    for video_path in df['path_to_video']:
        name = video_path.rsplit('\\')[-1].split('.')[0]
        cap = cv2.VideoCapture(video_path)
        count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # берем 5 равномерно распределенных кадров
        frames = np.linspace(1, count_frames, 5, dtype='int32')
        for n_frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
            ret, frame = cap.read()

            # выполняем повышение контрастности для удаления засвеченности
            frame_new = contrast_increase(frame)
            cv2.imwrite(os.path.join('../images_for_labeling', f'{name}_{n_frame}.jpg'), frame_new)


def save_train_test_df(path_to_video: str, random_state: int = 2023, test_ratio: float = 0.2):
    df = pd.DataFrame()
    path_list = []
    for entry in os.scandir(path_to_video):
        if entry.is_file() and entry.name.endswith('.mp4'):
            path_list.append(entry.path)
    df['path_to_video'] = path_list
    test = df.sample(frac=test_ratio, replace=False, random_state=random_state)
    train = df[~df.index.isin(test.index.values)]
    test.to_csv('../dataframes/train_out_test.csv', index=False)
    train.to_csv('../dataframes/train_out_train.csv', index=False)


if __name__ == '__main__':
    path_video_dir = '../prepair_dataset/train/train_in_out'
    save_train_test_df(path_to_video=path_video_dir)
    frames_for_labeling(path_to_df='../dataframes/train_out_train.csv')
