from get_problem_frames import contrast_increase
import cv2
from torchvision import datasets, transforms
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def get_one_embedding(module, args, output):
    """Вспомогательная функция, которая сохраняет состояние предпоследнего слоя ResNet (эмбеддинг)"""
    cur_emb = output[:, :, 0, 0].detach().numpy()
    embeddings.append(cur_emb)
    return None


def cv2_img_to_emb(img: np.ndarray):
    """Функция переводит из формата cv2 в формат torchTensor b выполняет инференс модели"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])(img).unsqueeze(
        0)  # size [1, 3, 256, 256]
    model(img_tensor)
    return None


def get_frames_and_emb(video: str):
    """
    Функция берет только средний кадр из видео и:
    1) Получает эмбеддинг фотографии
    2) Сохраняет картинку в отдельную директорию
    """
    file_name = video.split('.')[0]
    cap = cv2.VideoCapture(f'{path_to_video}/{name_class}/{video}')
    count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frame = count_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)

    ret, frame = cap.read()
    frame_new = contrast_increase(frame)
    cv2_img_to_emb(frame_new)
    cv2.imwrite(os.path.join(path_to_img_for_emb, f'{name_class}/{file_name}.jpg'), frame_new)


if __name__ == '__main__':
    path_to_video = '../prepair_dataset/train'
    path_to_img_for_emb = '../images_for_emb'

    print("Введите имя класса видео")
    name_class = input()
    video_list = [f for f in os.listdir(f'{path_to_video}/{name_class}') if f.endswith('.mp4')]
    embeddings = []

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    layer = model._modules.get('avgpool')  # предпоследний слой ResNet
    # добавляем функцию, которая будет вызываться при выполнении forward() после предпоследнего слоя
    _ = layer.register_forward_hook(get_one_embedding)
    model.eval()  # режим модели - inference

    for video in tqdm(video_list):
        get_frames_and_emb(video)

    embeddings = np.vstack(embeddings)  # shape [len(video_list), 512]
    matx = np.hstack((np.array(video_list).reshape(-1, 1), embeddings))
    emb_df = pd.DataFrame(matx).rename(columns={0: 'file_name'})
    emb_df.to_csv(f'../dataframes/{name_class}_emb.csv', index=False)
