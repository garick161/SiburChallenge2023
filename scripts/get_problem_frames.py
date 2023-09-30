from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os

df = pd.read_csv('../dataframes/false_class_df.csv')  # датасет с неверно классифицированными видео
model = YOLO('best.pt')  # предобученная модель с весам
path_to_images = '../bad_frames'


def contrast_increase(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    frame_new = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame_new


def save_problem_frame(video_path: str):
    """
    Функция отбирает по одному кадру в секунду, распознает объекты в кадре и сохраняет в файл с нанесением bounding_box
    :param video_path: str
    :return: None
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
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    im.save(os.path.join(path_to_images, f"{name_file}_{str(frame_id)}.jpg"))  # save image
        else:  # кадры закончились
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == '__main__':
    for video in df['path']:
        save_problem_frame(video)
