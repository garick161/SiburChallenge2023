from contrast_increase import contrast_increase
from ultralytics import YOLO
import numpy as np
import cv2


def detect_class(video_path: str, path_to_weights: str):
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

    model = YOLO(path_to_weights)
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
                results = model(frame_new, stream=True, conf=0.5)
                # results = model(frame_new, show=True)
                # создадим матрицу для записи результатов детекции классов на каждом кадре
                class_arr = np.zeros(shape=8, dtype='uint8')
                centr_xcoord_arr = np.zeros(shape=8)
                centr_ycoord_arr = np.zeros(shape=8)

                # Перебираем результаты
                for r in results:
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
                            if x1 > grad.shape[1] // 2 and y2 > grad.shape[
                                0] // 2:  # точка должна быть в нижнем правом квадрате
                                if y2 >= grad.shape[0]:
                                    y2 = grad.shape[0] - 1
                                while grad[y2][x1] == 0:  # если нижняя точка окна черная, значит попали в черную полосу
                                    # внизу кадра, которая на некоторых снимках присутствует
                                    y2 -= 10  # отступаем на 10 пкс вверх и еще раз проверяем

                                bridge_up_flag = True
                                check_movie_point_x = x1 - 100
                                check_movie_point_y = y2

                        # Если объект в кадре из списка ['plate_type_1', 'plate_type_2'],
                        # то фиксируем его координаты центров по x и по y для оценки движения в кадре в будущем
                        if cls in (5, 6):
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
            arr_coord = arr_coord[arr_coord > 0]
            # сделаем сдвиг массива и посчитаем разницу(разницу между соседними элементами)
            diff = arr_coord[1:] - arr_coord[:-1]
            # Если в массиве достаточно ненулевых элементов и элементы в массиве diff имеют одинаковый знак, значит
            # значение координат в исходном массиве либо монотонно возрастают или монотонно убывают. Следовательно,
            # есть движение вагона на видео
            if np.count_nonzero(diff) > len(diff) * 0.6:
                diff = diff[diff != 0]
                res = (diff > 0).all() if diff[0] > 0 else (diff < 0).all()
                return res

    # Матрица наличия классов на видео shape(count_frames, count_classes)
    class_matx = np.array(class_matx)

    # стандартные отклонения цвета пикселя в тестовом окне
    std_1 = np.std(pick_grad_list_1)
    std_2 = np.std(pick_grad_list_2)
    std_3 = np.std(pick_grad_list_3)
    median_point_std = np.median([std_1, std_2, std_3])

    # матрицы координат смещения объектов сверху вагона ['plate_type_1', 'plate_type_2']
    matx_xcoord = np.array(matx_xcoord)[:, [5, 6]]
    matx_ycoord = np.array(matx_ycoord)[:, [5, 6]]

    # Итоговое значение классов по всем кадрам: 1 - объект присутствует, 0 - нет. 1 только в том случае,
    # когда присутствует объект на 50% кадров и более
    result_class_arr = np.where(np.sum(class_matx, axis=0) > len(class_matx) // 2, 1, 0)

    # Принятие решение об операции на видео
    if result_class_arr[0] == 1 or result_class_arr[1] == 1:  # 'bridge_down_type_1', 'bridge_down_type_2'
        return 'bridge_down'

    elif np.sum(class_matx, axis=0)[4] > 1:  # 'coupling' более одного раза встречалось на видео
        return 'train_in_out'

    elif result_class_arr[7] == 1:  # track in frame
        return 'no_action'

    elif result_class_arr[2] == 1 or result_class_arr[3] == 1:  # bridge_up_type_1 or bridge_up_type_2 in frame
        if np.sum(class_matx, axis=0)[5] > len(class_matx) // 3 or np.sum(class_matx, axis=0)[6] > len(
                class_matx) // 3:  # plate_type_1, plate_type_2 in frame
            if check_move_wagon(matx_xcoord) or check_move_wagon(matx_ycoord):
                return 'train_in_out'
            elif median_point_std > 5:
                return 'train_in_out'
            else:
                return 'bridge_up'
        else:  # bridge_up_type_1' or 'bridge_up_type_2 and no one object detected else
            # Если на кадрах не обнаружилось площадок, попробуем оценить наличие движение по
            # тестовому окну слева внизу от поднятого мостика(изменение градиента цвета пикселей)
            if median_point_std < 5:
                return 'no_action'
            else:
                return 'train_in_out'
    else:  # no one object detected in whole frame
        #  Если ни одного объекта не обнаружено на всех кадрах, можно сделать такое предположение,
        #  что ведутся работы на крыше цистерны, много пара, поэтому ничего не видно
        return 'bridge_down'
