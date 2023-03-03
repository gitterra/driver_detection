from tools.yolo import yolo_ai
from tools import belt, driver, drowsiness, food, microsleep, phone, smoking, bcolors, blur
import gdown
import cv2
from IPython import display
import matplotlib.pyplot as plt

class Worker:
    drowsiness_model_url = 'https://storage.yandexcloud.net/terratraineeship/22_Kamaz/models/shape_predictor_68_face_landmarks.dat'

    def __init__(self):
      print(f'{bcolors.OKBLUE}Первоначальная настройка системы', end='')
      self.detector = yolo_ai.Detector()
      gdown.download(self.__class__.drowsiness_model_url, None, quiet=True)
      self.driver_identification = driver.Worker()
      self.belt_detector = belt.Worker()
      self.food_detector = food.Worker()
      self.phone_detector = phone.Worker()
      self.blur_detector = blur.Worker()
      self.smoking_detector = smoking.Worker()
      self.microsleep_detector = microsleep.Worker()
      self.drowsiness_detector = drowsiness.Worker()
      self.directory = 'tools/yolo/yolov5ai/yolov5/runs/detect/exp'
      print(f'{bcolors.ENDC}{bcolors.OKGREEN} Done{bcolors.ENDC}')    
    

    def video_process(self, path, start=None, stop=None):
      self.detector.run(path, start, stop)
      video = cv2.VideoCapture(path)
      self.fps = video.get(cv2.CAP_PROP_FPS)
      video.release()
      driver_name = self.driver_identification.identification(f'{self.directory}/crops/head')
      drowsiness_data = self.convert_to_interval(self.drowsiness_detector.analysis(f'{self.directory}/crops/head'))
      belt_data = self.convert_to_interval(self.belt_detector.analysis(f'{self.directory}/labels'))
      food_data = self.convert_to_interval(self.food_detector.analysis(f'{self.directory}/labels'))
      phone_data = self.convert_to_interval(self.phone_detector.analysis(f'{self.directory}/labels'))
      blur_data = self.convert_to_interval(self.blur_detector.analysis(f'{self.directory}/labels'))
      smoking_data = self.convert_to_interval(self.smoking_detector.analysis(f'{self.directory}/labels'))
      microsleep_data = self.convert_to_interval(self.microsleep_detector.analysis(f'{self.directory}/labels'))
      blur_data = self.convert_to_interval(self.blur_detector.analysis(f'{self.directory}/labels'))
      self.show_result((driver_name, drowsiness_data, belt_data, food_data, phone_data, smoking_data, microsleep_data, blur_data))

    def convert_to_interval(self, data):
      if data:
        frame_rate = self.fps
        intervals = []
        start = end = data[0]
        for frame_num in data[1:]:
            if frame_num == end + 1:
                end = frame_num
            else:
                intervals.append(((start-1)/frame_rate, end/frame_rate))
                start = end = frame_num
        intervals.append(((start-1)/frame_rate, end/frame_rate))
        intervals = [(round(interval[0], 1), round(interval[1], 1)) for interval in intervals]
        return intervals
      else:
        return [(0,0.01)]
    
    def show_result(self, data):
      print()
      print()
      print()
      print(f'{bcolors.HEADER}{bcolors.BOLD}Результат обработки видеофайла{bcolors.ENDC}')
      print(f'  {bcolors.OKCYAN}{bcolors.BOLD}Водитель:{bcolors.ENDC} {data[0]}')
      color_list = [
          '#FC67BD',
          '#B15DE3',
          '#7673F9',
          '#5DA9E3',
          '#4DFFE1',
          '#2D5BE3',
          '#DB2DE3'
          
      ]
      intervals = [('Сонливость', data[1]), 
                  ('Ремень', data[2]), 
                  ('Прием пищи', data[3]), 
                  ('Телефон', data[4]), 
                  ('Курение', data[5]), 
                  ('Микросон', data[6]), 
                  ('Размытое изображение', data[7])
                  ]
      
      interval_dict = {}

      for interval in intervals:
          if interval[0] not in interval_dict:
              interval_dict[interval[0]] = []
          interval_dict[interval[0]].extend(interval[1])

      # Define the categories for the y-axis
      categories = list(interval_dict.keys())

      # Create a figure and axis
      fig, ax = plt.subplots(figsize=(30, 7))

      # Set the y-ticks and labels
      ax.set_yticks(range(len(categories)))
      ax.set_yticklabels(categories)

      # Iterate through the intervals and add rectangles to the graph
      for i, category in enumerate(categories):
          for interval in interval_dict[category]:
              ax.add_patch(plt.Rectangle((interval[0], i-0.4), interval[1]-interval[0], 0.8, color=color_list[i]))

      # Add vertical lines every 10 seconds
      for i in range(int(max([interval[1] for intervals in interval_dict.values() for interval in intervals]))//10 + 1):
          ax.axvline(x=i*10, color='gray', linestyle='--')

      # Set the x-label and limits
      ax.set_xlabel('Time')
      ax.set_xlim(0, max([interval[1] for intervals in interval_dict.values() for interval in intervals]))

      # Add some extra space at the top and bottom of the graph
      ax.margins(y=0.1)

      # Show the graph
      plt.show()
      for key,value in intervals: 
        print(f'   {w.bcolors.HEADER}{w.bcolors.BOLD}{key}:{w.bcolors.ENDC}{value}') 
