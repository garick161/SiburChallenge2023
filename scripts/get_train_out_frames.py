import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import os
from contrast_increase import contrast_increase


def frames_for_labeling(path_to_df: str):
    """Функция для получения фреймов из видео для дальнейшей разметки"""
    df = pd.read_csv(path_to_df)

    for row in tqdm(range(len(df))):
        cap = cv2.VideoCapture(df['path_to_video'][row])
        count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_path = (df['path_to_img'][row]).rsplit('.', 1)
        # берем 5 равномерно распределенных кадров
        frames = np.linspace(1, count_frames, 5, dtype='int32')
        for n_frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
            ret, frame = cap.read()

            # выполняем повышение контрастности для удаления засвеченности
            frame_new = contrast_increase(frame)

            cv2.imwrite(f'_{n_frame}.'.join(img_path), frame_new)


def save_train_test_df(df: pd.DataFrame, random_state: int = 2023, test_ratio: float = 0.2):
    test = df.sample(frac=test_ratio, replace=False, random_state=random_state)
    train = df[~df.index.isin(test.index.values)]
    print(f'Всего видеофайлов: {len(df)}')
    print(f'train: {len(train)}')
    print(f'test: {len(test)}')
    test.to_csv('../dataframes/train_out_test.csv', index=False)
    train.to_csv('../dataframes/train_out_train.csv', index=False)


if __name__ == '__main__':
    path_to_video = '../prepair_dataset/train/train_in_out'
    path_to_save = '../images_for_labeling'
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    #  Подготовка структуры датафрейма
    df = pd.DataFrame()
    df['video_name'] = [f for f in os.listdir(path_to_video) if f.endswith('.mp4')]
    df['true_label'] = 'train_in_out'
    df['path_to_video'] = df['video_name'].apply(lambda name: f'{path_to_video}/{name}')
    df['path_to_img'] = df['video_name'].apply(lambda name: f"{path_to_save}/{name.split('.')[0]}.jpg")
    save_train_test_df(df=df)
    frames_for_labeling(path_to_df='../dataframes/train_out_train.csv')
