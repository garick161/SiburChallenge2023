from ultralytics import YOLO
import cv2
from PIL import Image
import os
from contrast_increase import contrast_increase


def save_problem_frame(video_path: str):
    """
    Функция отбирает по одному кадру в секунду, распознает объекты в кадре и сохраняет в файл с нанесением bounding_box
    :param video_path: str
    :return: None
    """
    print(video_path)
    name_file = video_path.rsplit('/', 1)[1].split('.')[0]
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
    model = YOLO('weights_ver4.pt')  # предобученная модель с весам
    path_to_images = '../bad_frames'
    save_problem_frame('../prepair_dataset/train/bridge_down/166d7e253ba332c3.mp4')
