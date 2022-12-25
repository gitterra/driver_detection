import pickle
import cv2
import os
import face_recognition
import gdown


class Worker:
    pre_encoder_url = 'https://storage.yandexcloud.net/aiueducation/KAMAZ/models/face_enc'
    pre_encoder_file = '/content/driver_detection/tools/face_enc'
    drivers = {
        '01_Igor': 'Игорь',
        '02_Ilsaf': 'Ильсаф',
        '03_Kamil': 'Камиль',
        '04_Boris': 'Борис',
        'Unknown': 'Unknown',
    }

    def __init__(self):
        if not os.path.exists(__class__.pre_encoder_file):
            gdown.download(
                __class__.pre_encoder_url,
                __class__.pre_encoder_file,
                quiet=True
            )
        self.data = pickle.loads(open(
            __class__.pre_encoder_file, "rb"
        ).read())

    def identification(self, directory, count_image_for_analysis=200):
        """
        Функция идентификации водителя

        @directory: директория, в которой хранятся изображения головы водителя
        @count_image_for_analysis:  количество изображений, которое будет анализироваться
                                  для идентификации водителя
        """

        # Проверка существования указанной директории
        if os.path.isdir(directory):
            counts = {}  # Словарь, который будет сичтать количество совпадений для каждого водителя
            cnt_find = 0
            # Берем первые count_image_for_analysis изображений
            for f in sorted(os.listdir(directory))[:count_image_for_analysis]:
                # Открытие и преобразование изображения
                image = cv2.imread(f'{directory}/{f}')
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Получение эмбеддинга открытого изображения
                encoding = face_recognition.face_encodings(rgb)

                # Если face_recognition не обнаружил лицо на изображении
                if not encoding:
                    continue
                cnt_find+=1
                if cnt_find>=20:
                    break
                # Сравнение экнодинга изображения с энкодингами в базе
                matches = face_recognition.compare_faces(self.data["encodings"], encoding[0])

                name = "Unknown"  # Имя водителя
                # Если сравнение энкодингов дало хотя бы одно совпадение
                if True in matches:
                    # Получаем индексы изображений в базе, которые совпали
                    matched_idxs = [i for (i, b) in enumerate(matches) if b]
                    # Для каждого индекса получаем имя водителя и вносим в словарь
                    for i in matched_idxs:
                        name = self.data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

            # Возвращаем имя водителя, который получил максимальное количество совпадений
            current_driver = max(counts, key=counts.get)

            if current_driver in __class__.drivers:
                return __class__.drivers[current_driver]
            else:
                if counts:
                    return max(counts, key=counts.get)
                else:
                    return 'Unknown'
        else:
            print(f'Директория {directory} не найдена')
