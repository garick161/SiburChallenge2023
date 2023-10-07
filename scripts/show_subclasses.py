import pandas as pd
import matplotlib.pyplot as plt
import cv2
from math import ceil


def plot_subclasses(path: str):
    """Функция для визуального анализа качества разбиения изображений на подклассы"""

    df = pd.read_csv(path)

    for sub_class in df['sub_class'].unique():
        temp_df = df[df['sub_class'] == sub_class]
        count_rows = ceil(len(temp_df) / 5)
        fig = plt.figure(figsize=(12, count_rows * 2))

        for i, name in enumerate(temp_df['path_to_img']):
            frame = cv2.imread(name)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fig.add_subplot(count_rows, 5, i + 1).set_title(f"sub_class: {sub_class}", size=9)
            plt.imshow(frame)
            plt.axis('off')

        plt.show()


if __name__ == '__main__':
    path_to_df = '../dataframes/bridge_down_with_subclass.csv'
    plot_subclasses(path_to_df)

