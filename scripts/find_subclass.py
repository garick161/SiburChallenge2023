import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

pd.options.mode.chained_assignment = None


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
    if (cosines < 0.15).all():  # Если все элементы очень похожи и разница не более 0.15 по сравнению с первым элементом
        # -> берем всех в один subclass
        return len(cosines)

    for i in range(1, len(cos_diff) - 1):
        if cosines[i] > 0.7:  # косинусное расстояние больше 1 => фреймы уже сильно различаются
            if frames_idx_list:
                return frames_idx_list[-1] + 1  # индекс последнего максимума
            else:
                return 1  # нет ни одного похожего фрейма
        if cos_diff[i - 1] > threshold or cos_diff[i] > threshold:  # значение выше уровня шума
            if (cos_diff[i] > cos_diff[i - 1]) and (
                    cos_diff[i] > cos_diff[i + 1]):  # обнаружен максимум последовательности
                if len(extremum_list) == 0:
                    extremum_list.append(cos_diff[i])
                    frames_idx_list.append(i)
                elif cos_diff[i] / extremum_list[-1] > 0.5:  # добавляем только те экстремумы, которые больше предыдущих
                    extremum_list.append(cos_diff[i])
                    frames_idx_list.append(i)
        if len(extremum_list) == sim_level:
            return i + 1

    # если заявленный уровень sim_level не достигнут, берем индекс последнего максимума
    try:
        return frames_idx_list[-1] + 1
    except IndexError as e:  # Если экстремумы не обнаружены, считаем все эти фреймы похожими
        return len(cosines)


def reduce_dim(df: pd.DataFrame, len_vect: int = 25) -> np.ndarray:
    pca = PCA(n_components=len_vect)
    return pca.fit_transform(df)


def plot_images(df: pd.DataFrame, split_idx: int):
    """Функция для изображения фреймов с косинусным расстояниями между ними от 1 фрейма"""
    count_rows = ceil(len(df) / 5)
    fig = plt.figure(figsize=(12, count_rows * 2))

    for i, img in enumerate(df['path_to_img']):
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if i < split_idx:
            color = 'black'
        else:
            color = 'red'
        fig.add_subplot(count_rows, 5, i + 1).set_title(f"idx: {i} / cosine: {round(df.loc[i]['cosine'], 3)}",
                                                        size=9, color=color)
        plt.imshow(frame)
        plt.axis('off')

    plt.show()


def plot_cosines(df: pd.DataFrame, split_idx: int):
    """Функция выводит 2 графика:
    1) Зависимость косинусного расстояния при переходе от 1-го фрейма к остальным
    2) Зависимость разницы косинусного расстояния при переходе от 1-го фрейма к остальным"""
    cosines = df['cosine'].values
    cos_diff = cosines[1:] - cosines[:-1]

    fig = plt.figure(figsize=(12, 5))
    ax = fig.subplots(nrows=1, ncols=2)

    ax[0].plot(cosines)
    ax[0].set_title('cosine')
    ax[0].set_xlabel('n_frames')
    if split_idx != len(df):
        ax[0].annotate('split_point', xy=(split_idx, cosines[split_idx]),
                       xytext=(split_idx + 1, cosines[split_idx] - 0.1),
                       arrowprops=dict(facecolor='red', shrink=0.05))
    ax[0].set_xticks(df.index)
    ax[0].set_yticks(np.arange(0, 1.1, 0.1))
    ax[0].grid()

    ax[1].plot(cos_diff)
    ax[1].set_title('cos_diff')
    ax[1].set_xlabel('n_frames')
    if split_idx != len(df):
        ax[1].annotate('split_point', xy=(split_idx - 1, cos_diff[split_idx - 1]),
                       xytext=(split_idx, cos_diff[split_idx - 1]),
                       arrowprops=dict(facecolor='red', shrink=0.05))
    ax[1].set_xticks(df.index)
    ax[1].set_yticks(np.arange(0, 0.75, 0.05))
    ax[1].axhline(y=0.05, color='red', linestyle='--')
    ax[1].grid()

    plt.tight_layout()
    plt.show()


def calc_cos_dist(df: pd.DataFrame, num_row: int) -> pd.DataFrame:
    """Функция подсчета косинусного расстояния"""
    matx = df.iloc[:, 4:].values
    df['cosine'] = cdist([matx[num_row]], matx, metric='cosine').flatten()
    df = df.sort_values('cosine').reset_index().rename(columns={'index': 'true_index'})
    return df


def demo_mode(path_to_df: str, search_range: int = 20, sim_level: int = 3):
    data = pd.read_csv(path_to_df)
    df = data.copy()
    df['sub_class'] = - 1
    num_row = np.random.choice(np.arange(len(df)))
    df = calc_cos_dist(df, num_row=num_row)
    split_idx = find_similar_frames(df.head(search_range)['cosine'].values, sim_level=sim_level)
    plot_images(df=df.head(search_range), split_idx=split_idx)
    plot_cosines(df=df.head(search_range), split_idx=split_idx)
    return None


def main_mode(path_to_df: str, search_range: int = 20, sim_level: int = 3, plot: bool = False):
    data = pd.read_csv(path_to_df)
    df = data.copy()
    df['sub_class'] = - 1
    for i in range(len(df)):
        temp_df = df[df['sub_class'] == -1]
        if len(temp_df) == 0:
            break
        temp_df = calc_cos_dist(temp_df, num_row=0)
        split_idx = find_similar_frames(temp_df.head(search_range)['cosine'].values, sim_level=sim_level)
        if plot:
            plot_images(df=temp_df.head(search_range), split_idx=split_idx)
            plot_cosines(df=temp_df.head(search_range), split_idx=split_idx)
        subclass_idxs = temp_df.head(split_idx)['true_index'].values
        df.loc[subclass_idxs, 'sub_class'] = i
    return df


if __name__ == '__main__':
    print('Введите имя класса видео')
    cls = input()
    path = f'../dataframes/{cls}_emb.csv'
    # demo_mode(path_to_df=path)

    df = main_mode(path_to_df=path, plot=False, sim_level=3, search_range=20)
    df.to_csv(f'../dataframes/{cls}_with_subclass.csv', index=False)
