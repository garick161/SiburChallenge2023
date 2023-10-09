from get_problem_frames import contrast_increase
import cv2
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.decomposition import PCA


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


def get_frames_and_emb(video_path: str, img_path: str):
    """
    Функция берет только средний кадр из видео и:
    1) Генерирует эмбеддинг фотографии
    2) Сохраняет картинку в отдельную директорию
    """
    cap = cv2.VideoCapture(video_path)
    count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frame = count_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)

    ret, frame = cap.read()
    frame_new = contrast_increase(frame)
    cv2_img_to_emb(frame_new)
    cv2.imwrite(img_path, frame_new)
    return None


def reduce_dim(matx: np.ndarray, len_vect: int = 25) -> np.ndarray:
    pca = PCA(n_components=len_vect)
    return pca.fit_transform(matx)


if __name__ == '__main__':
    print("Введите имя класса видео")
    cls = input()
    path_to_video = f'../prepair_dataset/train/{cls}'
    path_to_img_dir = f'../images_for_emb/{cls}'

    if not os.path.exists(path_to_img_dir):
        os.mkdir(path_to_img_dir)
    if not os.path.exists('../dataframes'):
        os.mkdir(path_to_img_dir)

    embeddings = []

    #  Подготовка структуры датафрейма
    df = pd.DataFrame()
    df['video_name'] = [f for f in os.listdir(path_to_video) if f.endswith('.mp4')]
    df['true_label'] = cls
    df['path_to_video'] = df['video_name'].apply(lambda name: f'{path_to_video}/{name}')
    df['path_to_img'] = df['video_name'].apply(lambda name: f"{path_to_img_dir}/{name.split('.')[0]}.jpg")

    # Загрузка модели
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    layer = model._modules.get('avgpool')  # предпоследний слой ResNet
    # добавляем функцию, которая будет вызываться при выполнении forward() после предпоследнего слоя
    _ = layer.register_forward_hook(get_one_embedding)
    model.eval()  # режим модели - inference

    # Перебираем все записи в датафрейме
    for row in tqdm(range(len(df))):
        get_frames_and_emb(video_path=df['path_to_video'][row], img_path=df['path_to_img'][row])

    embeddings = np.vstack(embeddings)
    embeddings = reduce_dim(matx=embeddings)  # снижаем размерность векторов до 25
    df = pd.concat((df, pd.DataFrame(embeddings)), axis=1)  # shape [count_videos, 29]

    df.to_csv(f'../dataframes/{cls}_emb.csv', index=False)
