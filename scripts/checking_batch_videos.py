from check_result_yolo8n import detect_class
import os
from loguru import logger
from datetime import datetime as dt
import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

classes = ['bridge_down_type_1', 'bridge_down_type_2', 'bridge_up', 'bridge_up_type_1', 'bridge_up_type_2',
           'coupling', 'plate_type_1', 'plate_type_2', 'track']
log_file_name = dt.now().strftime('%H-%M_%d-%m-%Y')
path_to_log_file = os.path.join('../logs', f'{log_file_name}.log')
logger.remove()  # Чтобы не выводилось в консоль, а писалось только в файл
logger.add(path_to_log_file, format='{time:HH:mm:ss} {message}')


df = pd.DataFrame()


class_list_detected = []
path_list = []


def help_func():
    count_video = 0
    count_class_detected = defaultdict(int)
    for entry in tqdm(os.scandir(path_to_dir)):
        if entry.is_file() and entry.name.endswith('.mp4'):
            count_video += 1
            logger.info('#######################################\n')
            logger.info(f'{entry.path}\n')
            start_time = time.time()
            stat_matx, result = detect_class(entry.path)
            global class_list_detected
            class_list_detected.append(result)
            global path_list
            path_list.append(entry.path)
            count_class_detected[result] += 1
            logger.info(f'Operation:{result}\n')
            logger.info("Process time: %s seconds\n#######################################\n\n" % (time.time() - start_time))
    else:
        logger.info(f'{count_class_detected}\ncount_video: {count_video}\n{dict(zip(classes, stat_matx))}')


if __name__ == '__main__':
    cls = input()
    path_to_dir = f'../prepair_dataset/train/{cls}'
    help_func()
    df['path'] = path_list
    df['predict'] = class_list_detected
    df['true_label'] = cls
    df.to_csv(f'../dataframes/{cls}_df.csv', index=False)
