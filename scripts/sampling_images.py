import shutil
import pandas as pd
import numpy as np
import os
from typing import List


def get_sample_indexes(path_to_df: str, random_state: int = 2023, reduce_coef: float = 1.0) -> List:
    """Функция определяет индексы фотографий, которые нужно взять для обучения
    """
    df = pd.read_csv(path_to_df)
    rs = np.random.RandomState(random_state)  # зафиксируем для воспроизводимости результатов

    # thresh - верхняя граница идеального равномерного распределения
    # При reduce_coef = 1 -> примерно 60% попадут в выборку
    # Если получается слишком много фотографий на ваш взгляд, то выборку можно уменьшить с помощью снижения reduce_coef
    # Мы будем использовать reduce_coef = 0.6
    thresh = int((len(df) / df['sub_class'].nunique()) * reduce_coef)
    print(f'Порог разбиения: {thresh} frames')
    idx_list = []

    for cls in df['sub_class'].unique():
        temp_df = df[df['sub_class'] == cls]
        if len(temp_df) <= thresh:
            idx_list += list(temp_df.index.values)
        else:
            idx_list += list(rs.choice(temp_df.index.values, thresh, replace=False))

    return idx_list


def copy_img_for_labeling(path_to_df: str, idx: List, path_to_save: str):
    df = pd.read_csv(path_to_df)

    files_list = df.loc[idx]['path_to_img'].to_list()
    for path in files_list:
        shutil.copy(os.path.join(path), os.path.join(path_to_save))
    return None


def save_train_test_df(path_to_df: str, idx: List, class_name: str):
    df = pd.read_csv(path_to_df)
    train = df[df.index.isin(idx)]
    test = df[~df.index.isin(idx)]
    print(f'Исходное количество фотографий: {len(df)}')
    print(f'Отобранное для обучения количество фотографий: {len(train)}')
    train.to_csv(f'../dataframes/{class_name}_train.csv', index=False)
    test.to_csv(f'../dataframes/{class_name}_test.csv', index=False)
    return None


if __name__ == '__main__':
    print('Введите имя класса видео')
    cls = input()
    path = f'../dataframes/{cls}_with_subclass.csv'
    path_to_save = '../images_for_labeling'
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    idx = get_sample_indexes(path_to_df=path, reduce_coef=0.6)
    copy_img_for_labeling(path_to_df=path, idx=idx, path_to_save=path_to_save)
    save_train_test_df(path_to_df=path, idx=idx, class_name=cls)
