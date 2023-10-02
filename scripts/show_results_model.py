from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os

from get_problem_frames import contrast_increase

df_bad = pd.read_csv('../dataframes/false_class_df.csv')  # датасет с неверно классифицированными видео
df_good = pd.read_csv('../dataframes/true_class_df.csv') # датасет с верно классифицированными видео
df = pd.concat((df_bad, df_good), ignore_index=True)
model = YOLO('weights_ver2.pt')  # предобученная модель с весам
path_to_images = '../frames_with_boundig_boxes'


def select_video(df: pd.DataFrame, count: int = 20) -> np.ndarray:
    """
    Функция выбирает по count видеороликов из каждого класса случайным образом
    """
    rs = np.random.RandomState(2023)  # зафиксируем random seed для воспроизводимости
    sel_video = np.array([])
    for cls in df['true_label'].unique():
        videos = df[df['true_label'] == cls]['path'].values
        size = count if len(videos) >= count else len(videos)
        sel_video = np.append(sel_video, rs.choice(a=videos, size=size, replace=False))

    return sel_video


def get_frames_with_bb(video_path: str):
    """
    Функция берет 1 кадр в секунду, производит детекцию объектов, наносит bounding_boxes
    """

    name_file = video_path.rsplit('\\', 1)[1].split('.')[0]
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if ret:
            frame_id = int(round(cap.get(1)))
            if frame_id % fps == 0:  # выбираем только каждый n-ый кадр
                # выполняем повышение контрастности для удаления засвеченности
                frame_new = contrast_increase(image=frame)

                # Работа с результатами модели YOLO
                results = model(frame_new)
                for r in results:
                    im_array = r.plot(pil=True)  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    im.save(os.path.join(path_to_images, f"{name_file}_{str(frame_id)}.jpg"))  # save image
        else:  # кадры закончились
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == '__main__':
    sel_videos = select_video(df=df, count=20)
    for path in sel_videos:
        get_frames_with_bb(path)
