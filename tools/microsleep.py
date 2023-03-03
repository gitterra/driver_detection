from tensorflow.keras.models import load_model
from imutils import face_utils
from scipy.spatial import distance as dist

import numpy as np

import os
import gdown
import cv2
import dlib
import re
from . import bcolors

class Worker:
  weights_file = '/content/yolo-master/shape_predictor_68_face_landmarks.dat'

  def __init__(self):
    if not os.path.exists(__class__.weights_file):
        print('Отсутствует файл весов')
  
  def microsleep(self, path, start=0, finish=-1, width=600):
    # определяем переменные
    EYE_AR_THRESH = 0.3          # Соотношение глаз
    EYE_AR_CONSEC_FRAMES = 90    # Порог мигания (в видео от КАМАЗ в секунде 30 кадров)
    COUNTER_flashing = 0         # Счетчик кадров для мигания
    COUNTER = 0                  # Счетчик кадров 
    mTOTAL = 0                   # Счетчик количества миганий
    filenames = []                                                              # определяем список с именами файлов в порядке возрастания
    result = {}                                                                 # определяем словарь для записи результата в формате: 'имя файла':  'результат определения лица'
    detector = dlib.get_frontal_face_detector()                                 # загружем детектор
    predictor = dlib.shape_predictor(__class__.weights_file)                                   # загружаем предиктор
    filenames = sorted(os.listdir(path), key= lambda x: os.path.getmtime(os.path.join(path, x)))
    result = dict.fromkeys(filenames, False)                                   

    # Получаем указатель на признаки лица левого и правого глаза и рта
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # идентифицируем точки 
    FACIAL_LANDMARKS_68_IDXS = dict([
      ("mouth", (48, 68)),
      ("right_eyebrow", (17, 22)),
      ("left_eyebrow", (22, 27)),
      ("right_eye", (36, 42)),
      ("left_eye", (42, 48)),
      ("nose", (27, 36)),
      ("jaw", (0, 17))
    ])

    # Функция определения открытого глаза
    def eye_aspect_ratio(eye):
      # Рассчитать расстояние по вертикали
      A = dist.euclidean(eye[1], eye[5])
      B = dist.euclidean(eye[2], eye[4])
      # Рассчитать расстояние по горизонтали
      C = dist.euclidean(eye[0], eye[3])
      ear = (A + B) / (2.0 * C)
      return ear  

    # функция получения координат
    def shape_to_np(shape, dtype="int"):
      # Создать 68 * 2
      coords = np.zeros((shape.num_parts, 2), dtype=dtype)
      # Обойти каждую ключевую точку
      # Получить координаты
      for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
      return coords
    for filename in filenames[start:finish]:                                    # выбираем диапазон
      data = cv2.imread(os.path.join(path,filename))                                      # считываем изображения по порядку
      if data is None:
        break
      #  resize и переводим в серый
      (h, w) = data.shape[:2]
      r = width / float(w)
      dim = (width, int(h * r))
      data = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)
      gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
      rects = detector(gray, 0)                                                 # Используем детектор для определения положения лица
      # Зацикливание информации о положении лица и использование предиктора для получения информации о положении лица
      for rect in rects:
        shape = predictor(gray, rect)                                           # Используем предиктор для определения положения лица
        shape = face_utils.shape_to_np(shape)                                   # Преобразование информации о чертах лица в формат массива
        leftEye = shape[lStart:lEnd]                                            # Извлекаем координаты левого и правого глаза
        rightEye = shape[rStart:rEnd]
        # Конструктор вычисляет значение EAR для левого и правого глаза, используя среднее значение в качестве окончательного EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        # Изображаем позицию глаз и рта
        leftEyeHull = cv2.convexHull(leftEye)        
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(data, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(data, [rightEyeHull], -1, (0, 255, 0), 1)
        # Делаем оценку для левого и правого глаза отдельно в качестве окончательной оценки. 
        # Если она меньше порога, добавляем 1, если она меньше порога в течение трех последовательных раз, это означает, что было выполнено одно событие мигания
        # Цикл, если условие выполнено, количество миганий +1
        if ear < EYE_AR_THRESH:                                                 # проверяем соотношение глаз
            COUNTER += 1
        else:
            # Если все три раза подряд меньше порога, это означает мигание
            if COUNTER >= EYE_AR_CONSEC_FRAMES:                                 # Сравниваем с установленным порогом
                mTOTAL += 1
                frameNumber = filenames.index(filename)                         # Получаем номер кадра в списке
                for f in filenames[(frameNumber - EYE_AR_CONSEC_FRAMES):(frameNumber + 1)]:
                    result[f] = True                                            # Записываем значение в словарь
            COUNTER = 0                                                         # Сброс счетчика кадров глаз 
    return  result, mTOTAL                                                      # Возвращаем номера изображений (result) и количество инцидентов (mTOTAL)

  def analisys(self, directory):
    print(f'{bcolors.OKBLUE}Этап: {bcolors.BOLD}МИКРОСОН', end='')
    try:
      result = []
      # контроль микросна  
      res, _ = self.microsleep(directory)
      for n_file in res.items(): 
        if n_file[1]:
          result.append(int(re.findall('[0-9]+', n_file[0])[0]))
      print(f'{bcolors.ENDC}{bcolors.OKGREEN} Done{bcolors.ENDC}')
      return result
    except:
      print(f'{bcolors.ENDC}{bcolors.FAIL} Done{bcolors.ENDC}')
      return []
