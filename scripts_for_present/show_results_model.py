from scripts.contrast_increase import contrast_increase
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import cv2
import os

model = YOLO('../scripts/weights_ver5.pt')  # предобученная модель с весам



def get_frames_with_bb(video_path: str, path_to_dir: str):
    """
    Функция берет 1 кадр в секунду, производит детекцию объектов, наносит bounding_boxes
    """
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)
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
                    im.save(f"{path_to_dir}/{name_file}_{str(frame_id)}.jpg")  # save image
        else:  # кадры закончились
            cv2.destroyAllWindows()
            cap.release()
            break


# cv2.rectangle(frame_new, (check_movie_point_x, check_movie_point_y - 20),
#               (check_movie_point_x + 60, check_movie_point_y), (255, 0, 0), 2)
# cv2.imshow("frame_new", frame_new)


if __name__ == '__main__':
    # sel_videos = select_video(df=df, count=20)
    # for path in sel_videos:
    path = '../frames_with_boundig_boxes/ver5'
    for entry in tqdm(os.scandir('../prepair_dataset/train/train_in_out')):
        if entry.is_file() and entry.name.endswith('.mp4'):
            get_frames_with_bb(video_path=entry.path, path_to_dir=path)
