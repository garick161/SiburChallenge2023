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
    if (cosines < 0.15).all():  # Если все элементы очень похожи и разница не более 0.1 по сравнению с первым элементом
                                # -> берем всех в один subclass
        return len(cosines)

    for i in range(1, len(cos_diff)):
        if cosines[i] > 1:  # косинусное расстояние больше 1 => фреймы уже сильно различаются
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
    except IndexError as e:
        print("Либо все фреймы одинаковы или очень близки, либо слишком высокий уровень 'sim_level'\nПопробуйте "
              "уменьшить значение 'sim_level'")
        return 19


def reduce_dim(df: pd.DataFrame, len_vect: int = 25) -> np.ndarray:
    pca = PCA(n_components=len_vect)
    return pca.fit_transform(df)


def plot_images(df: pd.DataFrame, split_idx: int):
    count_rows = ceil(len(df) / 5)
    fig = plt.figure(figsize=(12, count_rows * 2))

    for i, name in enumerate(df['file_name']):
        img_name = name.split('.')[0]
        frame = cv2.imread(f'../images_for_emb/no_action/{img_name}.jpg')
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
    ax[0].grid()
    ax[1].plot(cos_diff)
    ax[1].set_title('cos_diff')
    ax[1].set_xlabel('n_frames')
    if split_idx != len(df):
        ax[1].annotate('split_point', xy=(split_idx - 1, cos_diff[split_idx - 1]),
                       xytext=(split_idx, cos_diff[split_idx - 1]),
                       arrowprops=dict(facecolor='red', shrink=0.05))
    ax[1].set_xticks(df.index)
    ax[1].grid()
    plt.tight_layout()

    plt.show()


def calc_cos_dist(df: pd.DataFrame, num_row: int) -> pd.DataFrame:
    matx = df.iloc[:, :25].values
    df['cosine'] = cdist([matx[num_row]], matx, metric='cosine').flatten()
    df = df.sort_values('cosine').reset_index().rename(columns={'index': 'true_index'})
    return df


def demo_mode(path_to_df: str, search_range: int = 20, sim_level: int = 3):
    data = pd.read_csv(path_to_df)  # len(df.columns) = 513, df.columns[0] = 'file_name'
    df_slim = pd.DataFrame(reduce_dim(data.iloc[:, 1:]))  # shape(len(df), 25)
    df_slim['file_name'] = data['file_name']
    df_slim['sub_class'] = - 1
    num_row = np.random.choice(np.arange(len(df_slim)))
    df_slim = calc_cos_dist(df_slim, num_row=num_row)
    split_idx = find_similar_frames(df_slim.head(search_range)['cosine'].values, sim_level=sim_level)
    plot_images(df=df_slim.head(search_range), split_idx=split_idx)
    plot_cosines(df=df_slim.head(search_range), split_idx=split_idx)
    return None


def main_mode(path_to_df: str, search_range: int = 20, sim_level: int = 3, plot: bool = False):
    data = pd.read_csv(path_to_df)  # len(df.columns) = 513, df.columns[0] = 'file_name'
    df_slim = pd.DataFrame(reduce_dim(data.iloc[:, 1:]))  # shape(len(df), 25)
    df_slim['file_name'] = data['file_name']
    df_slim['sub_class'] = - 1
    for i in range(len(df_slim)):
        temp_df = df_slim[df_slim['sub_class'] == -1]
        if len(temp_df) == 0:
            break
        temp_df = calc_cos_dist(temp_df, num_row=0)
        split_idx = find_similar_frames(temp_df.head(search_range)['cosine'].values, sim_level=3)
        if plot:
            plot_images(df=temp_df.head(search_range), split_idx=split_idx)
            plot_cosines(df=temp_df.head(search_range), split_idx=split_idx)
        subclass_idxs = temp_df.head(split_idx)['true_index'].values
        df_slim.loc[subclass_idxs, 'sub_class'] = i
    return df_slim


if __name__ == '__main__':
    path = '../dataframes/no_action_emb.csv'
    # demo_mode(path_to_df=path)

    df = main_mode(path_to_df=path, plot=True, search_range=15)
    # df.to_csv('../dataframes/no_action_with_subclass.csv', index=False)


