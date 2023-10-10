from scripts.contrast_increase import contrast_increase
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import cv2
import os


def get_frames_with_bb(video_path: str, path_to_dir: str, path_to_weights: str):
    """
    Функция берет 1 кадр в секунду, производит детекцию объектов, наносит bounding_boxes
    """
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)
    name_file = video_path.rsplit('\\', 1)[1].split('.')[0]
    model = YOLO(path_to_weights)  # предобученная модель с весам
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    bridge_up_flag = False
    # координаты нижней левой точки контрольного окна для анализа градиента цвета
    check_movie_point_x = 0
    check_movie_point_y = 0

    while True:
        ret, frame = cap.read()
        if ret:
            frame_id = int(round(cap.get(1)))
            if frame_id % fps == 0:  # выбираем только каждый n-ый кадр
                # выполняем повышение контрастности для удаления засвеченности
                frame_new = contrast_increase(image=frame)
                class_arr = np.zeros(shape=8, dtype='uint8')

                # Работа с результатами модели YOLO
                results = model(frame_new)
                for r in results:
                    im_array = r.plot(pil=True)  # plot a BGR numpy array of predictions
                    grad = cv2.cvtColor(im_array, cv2.COLOR_BGR2GRAY)

                for r in results:
                    im_array = r.plot(pil=True)
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        class_arr[cls] = 1  # класс присутствует на слайде -> добавляем в матрицу 1 по индексу класса
                        # нахождение координат окна для анализа движения в кадре
                        if not bridge_up_flag and cls in (2, 3):
                            # Если cls in (0, 1) (bridge_down) нет смысла искать движение
                            # Если cls in (2, 3) (bridge_up) находим координаты только один раз, чтобы зафиксировать точку окна
                            # Оптимальное положение левее и ниже от левой нижней точки bounding_box 'bridge_up'
                            x1, _, _, y2 = box.xyxy[0]
                            x1 = int(x1)
                            y2 = int(y2)
                            if x1 > im_array.shape[1] // 2 and y2 > im_array.shape[
                                0] // 2:  # точка должна быть в нижнем правом квадрате
                                if y2 >= im_array.shape[0]:
                                    y2 = im_array.shape[0] - 1

                                while grad[y2][x1] == 0:  # если нижняя точка окна черная, значит попали в черную полосу
                                    # внизу кадра, которая на некоторых снимках присутствует
                                    y2 -= 10  # отступаем на 10 пкс вверх и еще раз проверяем
                                bridge_up_flag = True
                                check_movie_point_x = x1 - 100
                                check_movie_point_y = y2
                if bridge_up_flag:
                    cv2.rectangle(im_array, (check_movie_point_x, check_movie_point_y - 20),
                                  (check_movie_point_x + 60, check_movie_point_y), (255, 0, 0), 2)
                cv2.imwrite(f"{path_to_dir}/{name_file}_{str(frame_id)}.jpg", im_array)

        else:  # кадры закончились
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == '__main__':
    # sel_videos = select_video(df=df, count=20)
    # for path in sel_videos:
    path = '../frames_with_boundig_boxes/ver5'
    weights_path = '../scripts/weights_ver5.pt'
    for entry in tqdm(os.scandir('../prepair_dataset/train/train_in_out')):
        if entry.is_file() and entry.name.endswith('.mp4'):
            get_frames_with_bb(video_path=entry.path, path_to_dir=path, path_to_weights=weights_path)
