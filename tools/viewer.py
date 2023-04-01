import os
import re
import dlib
import cv2
from . import bcolors


class Worker:
    def __init__(self):
        self.index_class = 2

    def analysis(self, directory):
        print(f'{bcolors.OKBLUE}Этап: {bcolors.BOLD}ОТВЛЕЧЕНИЕ ВЗГЛЯДА', end='')
        try:
            result = []
            detector = dlib.get_frontal_face_detector() 
            if os.path.isdir(directory):
                for f in sorted(os.listdir(directory)):
                    data = cv2.imread(f'{directory}/{f}')      
                    if data is None:
                      break
                    (h, w) = data.shape[:2]   
                    r = width / float(w)
                    dim = (width, int(h * r))
                    data = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)              # меняем размер фото
                    face = detector(data,0) 
                    result.append(int(re.findall('[0-9]+', f)[0]))
            print(f'{bcolors.ENDC}{bcolors.OKGREEN} Done{bcolors.ENDC}')
            return sorted(result)
        except:
            print(f'{bcolors.ENDC}{bcolors.FAIL} Done{bcolors.ENDC}')
            return []
