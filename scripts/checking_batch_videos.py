from check_result_model_with_logs import detect_class
from collections import defaultdict
from datetime import datetime as dt
import time
from loguru import logger
from tqdm import tqdm
import os


def check_video(path_to_weights: str, path_to_dir: str):
    """
    Функция выполняет запускает определение классов для массива из видеофайлов:
    - перебивает все файлы по указанному пути и запускает процесс определения класса.
    Используется функция "detect_class" из файла "check_result_model_with_logs.py"
    - собирает статистику
    - администрирует логирование

    :param path_to_weights: Путь к весам модели YOLO 8n
    :param path_to_dir: Путь к директории с файлами
    :return: None
    """
    count_video = 0
    count_class_detected = defaultdict(int)
    for entry in tqdm(os.scandir(path_to_dir)):
        if entry.is_file() and entry.name.endswith('.mp4'):
            count_video += 1
            logger.info('#######################################\n')
            logger.info(f'{entry.path}\n')
            start_time = time.time()
            stat_matx, result = detect_class(entry.path, path_to_weights)  # Результат работы модели
            count_class_detected[result] += 1
            logger.info(f'Operation:{result}\n')
            logger.info(
                "Process time: %s seconds\n#######################################\n\n" % (time.time() - start_time))
    else:
        logger.info(f'{count_class_detected}\ncount_video: {count_video}\n{dict(zip(classes, stat_matx))}')

    return None


if __name__ == '__main__':
    print("Введите имя тестируемого класса")
    cls = input()

    weights_path = '../weights/weights_ver5.pt'
    dir_path = f'../prepair_dataset/train/{cls}'
    path_to_logs = '../logs'
    if not os.path.exists(path_to_logs):
        os.mkdir(path_to_logs)

    #  настройка файла логирования
    classes = ['bridge_down_1', 'bridge_down_2', 'bridge_up_1', 'bridge_up_2', 'coupling', 'plate_type_1',
               'plate_type_2', 'track']
    log_file_name = dt.now().strftime('%H-%M_%d-%m-%Y')
    path_to_log_file = f'{path_to_logs}/{log_file_name}.log'
    logger.remove()  # Чтобы не выводилось в консоль, а писалось только в файл
    logger.add(path_to_log_file, format='{time:HH:mm:ss} {message}')
    logger.info(f'Weight_of_model: {weights_path}')

    check_video(path_to_weights=weights_path, path_to_dir=dir_path)
