from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from get_problem_frames import contrast_increase


class_detect_stats = np.zeros(shape=12, dtype='int32')  # для анализа какие классы лучше определяются


@logger.catch
def detect_class(video_path):
    classes = ['bridge_down_type_1', 'bridge_down_type_2', 'bridge_up', 'bridge_up_type_1', 'bridge_up_type_2',
               'coupling', 'plate_type_1', 'plate_type_2', 'track']

    # Подготовим нужные объекты
    # Списки для формирования матриц градиента цвета в контрольных точках
    pick_grad_list_1 = []
    pick_grad_list_2 = []
    pick_grad_list_3 = []
    # Матрица регистрации объектов в каждом кадре(будут состоять из 0 и 1, где 1-был объект в кадре, 0- нет
    class_matx = []
    # Матрицы смещения координат центров движущихся объектов
    matx_xcoord = []
    matx_ycoord = []

    bridge_up_flag = False
    # координаты нижней левой точки контрольного окна для анализа градиента цвета
    check_movie_point_x = 0
    check_movie_point_y = 0

    model = YOLO('weights_ver2.pt')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if ret:
            frameId = int(round(cap.get(1)))
            if frameId % fps == 0:  # выбираем только кадр в секунду
                # выполняем повышение контрастности для удаления засвеченности
                frame_new = contrast_increase(frame)
                # полотно в оттенках серого для определения движения по контрасту пикселей
                grey = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
                grad = cv2.GaussianBlur(grey, (5, 5), 0)

                # Работа с результатами модели YOLO
                # results = model(frame_new, stream=True)
                results = model(frame_new, show=True)
                # создадим матрицу для записи результатов детекции классов на каждом кадре
                class_arr = np.zeros(shape=9, dtype='uint8')
                centr_xcoord_arr = np.zeros(shape=9)
                centr_ycoord_arr = np.zeros(shape=9)

                # Перебираем результаты
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        class_arr[cls] = 1  # класс присутствует на слайде -> добавляем в матрицу 1 по индексу класса
                        # нахождение координат окна для анализа движения в кадре
                        if not bridge_up_flag and cls in (3, 4):
                            # Если cls in (0, 1) (bridge_down) нет смысла искать движение
                            # Если cls in (3, 4) (bridge_up) находим координаты только один раз, чтобы зафиксировать точку окна
                            # Оптимальное положение левее и ниже от левой нижней точки bounding_box 'bridge_up'
                            x1, _, _, y2 = box.xyxy[0]
                            x1 = int(x1)
                            y2 = int(y2)
                            if (y2 + 60) < frame_new.shape[0]:  # Если не выходит за нижние границы изображения
                                y2 += 50
                            else:
                                y2 -= 10  # отступ 10 пкс, чтобы не попасть в черную полосу внизу кадра,
                                # которая на некоторых снимках присутствует
                            bridge_up_flag = True
                            check_movie_point_x = x1 - 100
                            check_movie_point_y = y2
                        # Если объект в кадре из списка ['plate_type_1', 'plate_type_2'],
                        # то фиксируем его координаты по x и по y для оценки движения в кадре в будущем
                        if cls in (6, 7):
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            centr_xcoord_arr[cls] = int(x1 + (x2 - x1) / 2)
                            centr_ycoord_arr[cls] = int(y1 + (y2 - y1) / 2)

                matx_xcoord.append(centr_xcoord_arr)
                matx_ycoord.append(centr_ycoord_arr)
                class_matx.append(class_arr)  # добавляем результаты кадра в матрицу

                if bridge_up_flag:
                    # Оцениваем значения 3 пикселей под 45 градусов. Затем считаем std. Если std > определенного
                    # значения => есть движение в кадре
                    # это не основной, а дополнительный метод определения движения
                    pick_grad_list_1.append(grad[check_movie_point_y][check_movie_point_x])
                    pick_grad_list_2.append(grad[check_movie_point_y - 10][check_movie_point_x + 30])
                    pick_grad_list_3.append(grad[check_movie_point_y - 20][check_movie_point_x + 60])
                    # cv2.rectangle(frame_new, (check_movie_point_x, check_movie_point_y - 20),
                    #               (check_movie_point_x + 60, check_movie_point_y), (255, 0, 0), 2)
                    # cv2.imshow("frame_new", frame_new)
        else:  # кадры закончились
            cv2.destroyAllWindows()
            cap.release()
            break

    def check_move_wagon(matx):
        # matx - матрица координат центров bounding_box для plate первого и второго типа.
        # shape(n, m), где n - количество кадров, m = 2(первый/второй тип площадки)
        non_zero = np.count_nonzero(matx, axis=0)
        if np.max(non_zero) != 0:  # Если ни одного объекта не обнаружено, нечего продолжать
            # Возьмем для анализа лишь тот объект, который больше всего представлен на слайдах
            arr_coord = np.array(matx[:, non_zero.argmax()])
            # сделаем сдвиг массива и посчитаем разницу(разницу между соседними элементами)
            diff = arr_coord[1:] - arr_coord[:-1]
            # Если в массиве достаточно ненулевых элементов и элементы в массиве diff имеют одинаковый знак, значит
            # значение координат в исходном массиве либо монотонно возрастают или монотонно убывают. Следовательно,
            # есть движение вагона на видео
            if np.count_nonzero(diff) > len(diff) // 2:
                diff = diff[diff != 0]
                return (diff > 0).all() if diff[0] > 0 else (diff < 0).all()

    # Матрица наличия классов на видео shape(count_frames, count_classes)
    class_matx = np.array(class_matx)
    logger.info(f'Матрица наличия классов на видео\n{pd.DataFrame(class_matx, columns=classes)}')

    # стандартные отклонения цвета пикселя в тестовом окне
    std_1 = np.std(pick_grad_list_1)
    std_2 = np.std(pick_grad_list_2)
    std_3 = np.std(pick_grad_list_3)
    median_point_std = np.median([std_1, std_2, std_3])
    logger.info(
        f'стандартные отклонения цвета пикселя в тестовом окне\nstd_1: {std_1}, std_2: {std_2}, std_3: {std_3}, median_point_std: {median_point_std}')

    # матрицы координат смещения объектов сверху вагона ['plate_type_1', 'plate_type_2']
    matx_xcoord = np.array(matx_xcoord)[:, [6, 7]]
    matx_ycoord = np.array(matx_ycoord)[:, [6, 7]]
    logger.info('матрицы координат смещения площадок сверху вагона')
    logger.info(
        f"Смещение по горизонтали\n{pd.DataFrame(matx_xcoord, columns=['plate_type_1', 'plate_type_2'])}")
    logger.info(
        f"Смещение по вертикали\n{pd.DataFrame(matx_ycoord, columns=['plate_type_1', 'plate_type_2'])}")

    # Итоговое значение классов по всем кадрам: 1 - объект присутствует, 0 - нет. 1 только в том случае,
    # когда присутствует объект на 50% кадров и более
    result_class_arr = np.where(np.sum(class_matx, axis=0) / len(class_matx) > len(class_matx) // 2, 1, 0)
    global class_detect_stats
    class_detect_stats += result_class_arr  # для логирования и статистики

    # Принятие решение об операции на видео
    if result_class_arr[0] == 1 or result_class_arr[1] == 1:  # 'bridge_down_type_1', 'bridge_down_type_2'
        logger.info('result_class_arr[0] == 1 or result_class_arr[1] == 1 => bridge_down')
        return class_detect_stats, 'bridge_down'

    elif np.sum(class_matx, axis=0)[5] > 1:  # 'coupling' более одного раза встречалось на видео
        logger.info('coupling more than one time in frame => train_in_out')
        return class_detect_stats, 'train_in_out'

    elif result_class_arr[8] == 1:  # track in frame
        logger.info('result_class_arr[8] == 1 free track detected => no_action')
        return class_detect_stats, 'no_action'

    elif result_class_arr[3] == 1 or result_class_arr[4] == 1:  # bridge_up_type_1 or bridge_up_type_2 in frame
        if result_class_arr[6] == 1 or result_class_arr[7] == 1:  # plate_type_1, plate_type_2 in frame
            if check_move_wagon(matx_xcoord) or check_move_wagon(matx_ycoord):
                logger.info(
                    'result_class_arr[3] or result_class_arr[4] -> plates detected -> move detected => train_in_out')
                return class_detect_stats, 'train_in_out'
            else:
                logger.info(
                    'result_class_arr[3] or result_class_arr[4] -> plates detected -> move not detected => bridge_up')
                return class_detect_stats, 'bridge_up'
        else:  # bridge_up_type_1' or 'bridge_up_type_2 and no one object detected else
            # Если на кадрах не обнаружилось площадок, попробуем оценить наличие движение по
            # тестовому окну слева внизу от поднятого мостика(изменение градиента цвета пикселей)
            if median_point_std < 5:
                logger.info(
                    'bridge_up_type_1 or bridge_up_type_2 and plates on top of wagon not detected -> check test wimdow -> median_std_point < 5 => no_action')
                return class_detect_stats, 'no_action'
            else:
                logger.info(
                    'bridge_up_type_1 or bridge_up_type_2 and plates on top of wagon not detected -> check test wimdow -> median_std_point > 5 => train_in_out')
                return class_detect_stats, 'train_in_out'
    else:  # no one object detected in whole frame
        #  Если ни одного объекта не обнаружено на всех кадрах, можно сделать такое предположение,
        #  что ведутся работы на крыше цистерны, много пара, поэтому ничего не видно
        logger.info(
            'no one object detected in whole frame -> work on wagon => bridge_down')
        return class_detect_stats, 'bridge_down'
