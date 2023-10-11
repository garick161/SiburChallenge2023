import numpy as np
import cv2
import os


def video2frame(path_video: str, path_to_dir: str, prefix_name: int, class_name: str):
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)

    cap = cv2.VideoCapture(path_video)
    count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # берем 10 равномерно распределенных кадров
    frames = np.linspace(1, count_frames, 10, dtype='int32')
    for i in range(len(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i] - 1)
        ret, frame = cap.read()
        cv2.rectangle(frame, (0,0), (100, 20), (0, 255, 0), -1)
        cv2.putText(frame, class_name, (2, 16), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(f'{path_to_dir}/{str(prefix_name)}{str(i)}.jpg', frame)


if __name__ == '__main__':
    cls = 'no_action'
    video_path = f'../sel_for_present/{cls}'
    dir_path = f'../sel_for_present/{cls}/images'
for i, entry in enumerate(os.scandir(video_path)):
    if entry.is_file() and entry.name.endswith('.mp4'):
        video2frame(path_video=entry.path, path_to_dir=dir_path, prefix_name=i, class_name=cls)
