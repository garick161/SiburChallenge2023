import numpy as np
import cv2
import os


def make_panorama(frame_1: str, frame_2: str, frame_3: str, frame_4: str, save_path: str, name: str):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    frame_1 = cv2.imread(frame_1)
    frame_2 = cv2.imread(frame_2)
    frame_3 = cv2.imread(frame_3)
    frame_4 = cv2.imread(frame_4)
    panorama = np.concatenate((frame_1, frame_2, frame_3, frame_4), axis=1)
    cv2.imwrite(f'{save_path}/{name}', panorama)


if __name__ == '__main__':
    img_names = os.listdir('../sel_for_present/bridge_up/images')
    path_to_save = '../sel_for_present/panoramas'
    for name in img_names:
        fr = '../se'
        make_panorama(frame_1=f'../sel_for_present/bridge_down/images/{name}',
                      frame_2=f'../sel_for_present/bridge_up/images/{name}',
                      frame_3=f'../sel_for_present/no_action/images/{name}',
                      frame_4=f'../sel_for_present/train_in_out/images/{name}',
                      save_path=path_to_save,
                      name=name)