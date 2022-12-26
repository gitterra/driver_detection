from tensorflow.keras.models import load_model
from imutils import face_utils

import numpy as np

import os
import gdown
import cv2
import dlib
import re

class Drowsiness:
  weights_url = 'https://storage.yandexcloud.net/aiueducation/KAMAZ/models/face_enc'
  weights_file = '/content/yolo-master/shape_predictor_68_face_landmarks.dat'
    
  def __init__(self):
    if not os.path.exists(__class__.weights_file):
        print('Отсутствует файл весов')

  def yawning(self, path, start=0, finish=-1, width=400):
    # определяем переменные
    MAR_THRESH = 0.5                                                            # Соотношение рта (зевание)
    MOUTH_AR_CONSEC_FRAMES = 90                                                 # Порог зевания
    mCOUNTER = 0                                                                # Счетчик кадров для зевания
    mTOTAL = 0                                                                  # Счетчик общего количества зевания
    filenames = []                                                              # определяем список с именами файлов в порядке возрастания
    result = {}                                                                 # определяем словарь для записи результата в формате: 'имя файла':  'результат определения лица'
    detector = dlib.get_frontal_face_detector()                                 # загружем детектор
    predictor = dlib.shape_predictor(__class__.weights_file)                                   # загружаем предиктор
    filenames = sorted(os.listdir(path), key= lambda x: os.path.getmtime(os.path.join(path, x)))
    result = dict.fromkeys(filenames, False)
    # Получаем указатель на признаки лица левого и правого глаза и рта
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
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
    # функция получения координат
    def shape_to_np(shape, dtype="int"):
    # Создать 68 * 2
      coords = np.zeros((shape.num_parts, 2), dtype=dtype)
      # Обойти каждую ключевую точку
      # Получить координаты
      for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
      return coords

    # функция определения открытого рта
    def mouth_aspect_ratio(mouth):
      A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
      B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
      C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
      mar = (A + B) / (2.0 * C)
      return mar

    for filename in filenames[start:finish]: 
      data = cv2.imread(os.path.join(path,filename))
      if data is None:
        break
      (h, w) = data.shape[:2]
      r = width / float(w)
      dim = (width, int(h * r))
      data = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)
      gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
      rects = detector(gray, 0)                                                 # Используем детектор для определения положения лица
    
      for rect in rects:
        shape = predictor(gray, rect)                                           # Используем предиктор для определения положения лица
        shape = face_utils.shape_to_np(shape)                                   # Преобразование информации о чертах лица в формат массива
        mouth = shape[mStart:mEnd]                                              # Координаты рта
        mar = mouth_aspect_ratio(mouth)                                         # Определяем зевок
        if mar > MAR_THRESH:                           # проверяем соотношение рта (определяем зевание)
          mCOUNTER += 1  
        else:
          if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:     # Сравниваем с установленным порогом
            mTOTAL += 1
            frameNumber = filenames.index(filename)                         # Получаем номер кадра в списке
            for f in filenames[(frameNumber - MOUTH_AR_CONSEC_FRAMES):(frameNumber + 1)]:
              result[f] = True 
          mCOUNTER = 0
    return  result, mTOTAL                                                      # Возвращаем  номера изображений (result) и количество инцидентов (mTOTAL)

  def analisys(self, directory):
    result=[]
    res, _ = self.yawning(directory, finish=len(os.listdir(directory)))
    for n_file in res.items(): 
      if n_file[1]:
        result.append(int(re.findall('[0-9]+', n_file[0])[0]))
    return result