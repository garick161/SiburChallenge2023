from get_problem_frames import contrast_increase
import cv2
import pandas as pd
import numpy as np
import os



# def save_frames(video_path):
#     """
#     Функция для получения фреймов из видео для дальнейшей разметки
#     :param video_path: str
#     :return: None
#     """
#     path, file = video_path.rsplit('\\', 1)
#     name_file = file.split('.')[0]
#     class_name = path.rsplit('/', 1)[1]
#
#     cap = cv2.VideoCapture(video_path)
#     count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # Если класс видео ('bridge_down', 'bridge_up', 'no_action') => берем только один центральный кадр
#     if class_name in ('bridge_down', 'bridge_up', 'no_action'):
#         n_frame = count_frames // 2
#         cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
#         ret, frame = cap.read()
#
#         # выполняем повышение контрастности для удаления засвеченности
#         frame_new = contrast_increase(frame)
#         cv2.imwrite(os.path.join(path_to_images, f'{name_file}_{n_frame}.jpg'), frame_new)
#     else:
#         # Если класс видео 'train_in_out' => берем 5 равномерно распределенных кадров
#         frames = np.linspace(1, count_frames, 5, dtype='int32')
#         for n_frame in frames:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
#             ret, frame = cap.read()
#
#             # выполняем повышение контрастности для удаления засвеченности
#             frame_new = contrast_increase(frame)
#             cv2.imwrite(os.path.join(path_to_images, f'{name_file}_{n_frame}.jpg'), frame_new)



def save_train_test_df(path_to_video: str, random_state: int=2023, test_ratio: float=0.2):
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
