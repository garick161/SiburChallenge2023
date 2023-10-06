import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from typing import List


def find_similar_frames(cosines: np.ndarray, sim_level: int = 3, threshold: float = 0.05) -> int:
    """
    Функция получает отсортированный массив косинусных расстояний по возрастанию и определяет индекс элемента.
    По этому индексу дальше будут разделяться картинки:
    - до этого индекса - фреймы похожи между собой и образуют свой subclass
    - индекс и все элементы после - совсем другие картинки.

    :param cosines: np.ndarray
    Отсортированный массив косинусных расстояний по возрастанию.

    :param sim_level: int default = 3
    Заданный уровень похожести [1, ...]. Чем ниже, тем более жесткий критерий отбора. То есть при
    sim_level = 1 даже идентичные картинки, с чуть разным освещением будет считаться разными. Чем больше sim_level, тем
    более лояльно будет оцениваться. sim_level = 3 - наиболее оптимальный вариант.

    :param threshold: float default = 3
    Уровень шума изменения косинусного расстояния. Ниже которого все значения будут игнорироваться.

    :return: int
    Индекс элемента по которому нужно делать разбиение
    """
    cos_diff = cosines[1:] - cosines[:-1]
    extremum_list = []
    frames_idx_list = []
    for i in range(1, len(cos_diff)):
        if cosines[i] > 1:  # косинусное расстояние больше 1 => фреймы уже сильно различаются
            if frames_idx_list:
                return frames_idx_list[-1] + 1  # индекс последнего максимума
            else:
                return 1  # нет ни одного похожего фрейма
        if cos_diff[i - 1] > threshold or cos_diff[i] > threshold:  # значение выше уровня шума
            if (cos_diff[i] > cos_diff[i - 1]) and (cos_diff[i] > cos_diff[i + 1]):  # обнаружен максимум последовательности
                if len(extremum_list) == 0:
                    extremum_list.append(cos_diff[i])
                    frames_idx_list.append(i)
                elif cos_diff[i] > np.max(extremum_list):  # добавляем только те экстремумы, которые больше предыдущих
                    extremum_list.append(cos_diff[i])
                    frames_idx_list.append(i)
        if len(extremum_list) == sim_level:
            return i + 1

    # если заявленный уровень sim_level не достигнут, берем индекс последнего максимума
    try:
        frames_idx_list[-1] + 1
    except IndexError as e:
        print("Либо все фреймы одинаковы или очень близки, либо слишком высокий уровень 'sim_level'\nПопробуйте "
              "уменьшить значение 'sim_level'")


def reduce_dim(df: pd.DataFrame, len_vect: int = 25) -> np.ndarray:
    pca = PCA(n_components=len_vect)
    return pca.fit_transform(df)


def find_subclass_idx(df, num_row):
    matx = df.iloc[:, :25].values
    df['cosine'] = cdist([matx[num_row]], matx, metric='cosine').flatten()
    df = df.sort_values('cosine').reset_index().rename(columns={'index': 'true_index'})
    split_idx = find_similar_frames(df.head(20)['cosine'].values)
    print(split_idx)
    res = df.loc[0:split_idx]['true_index'].values
    print(res)
    return res


if __name__ == '__main__':
    mode_dict = {1: 'demo_mode', 2: 'refactoring_mode', 3: 'main_mode'}
    print('Выберите режим работы\n1 - demo_mode\t2 - refactoring_mode\t3 - main_mode\nВведите только цифру')
    mode = mode_dict[int(input())]
    data = pd.read_csv('../dataframes/no_action_emb.csv')  # len(df.columns) = 513, df.columns[0] = 'file_name'
    df_slim = pd.DataFrame(reduce_dim(data.iloc[:, 1:]))  # shape(len(df), 25)
    df_slim['file_name'] = data['file_name']
    df_slim['sub_class'] = - 1

    for i in range(len(df_slim)):
        temp_df = df_slim.copy()
        subclass_idx = find_subclass_idx(df=temp_df, num_row=i)
        df_slim.loc[subclass_idx, 'sub_class'] = i
        print(df_slim['sub_class'].value_counts())
        break