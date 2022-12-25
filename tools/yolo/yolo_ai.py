from TerraYoloTest.TerraYolo import TerraYoloV5
from tools import bcolors
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import gdown
import shutil


class Detector:
    TRAIN_DIR = '/content/driver_detection/tools/yolo/yolov5ai/'
    url_weights = 'https://storage.yandexcloud.net/aiueducation/KAMAZ/models/yolo_from_framework/best_50.pt'
    weights = os.path.join(TRAIN_DIR, 'best.pt')

    def __init__(self):
        if not os.path.exists(__class__.TRAIN_DIR):
            os.mkdir(__class__.TRAIN_DIR)
        print(f'{bcolors.OKBLUE}Первоначальная настройка системы', end='')
        self.model = TerraYoloV5(
            work_dir=__class__.TRAIN_DIR
        )
        self.load_weights()
        print(f'{bcolors.ENDC}{bcolors.OKGREEN} Done{bcolors.ENDC}')

    @staticmethod
    def load_weights():
        gdown.download(__class__.url_weights, __class__.weights, quiet=True)

    @staticmethod
    def get_work_dir():
        path = os.path.join(__class__.TRAIN_DIR, 'yolov5')
        return path

    def run(
            self,
            f_name,
            start=None,
            stop=None,
            save_labels=True,
            save_crop=True,
            conf=0.5
    ):
        if os.path.exists(os.path.join(
          __class__.TRAIN_DIR,
          'yolov5/runs/detect')):
            shutil.rmtree(os.path.join(
              __class__.TRAIN_DIR,
              'yolov5/runs/detect'))
            os.mkdir(os.path.join(
              __class__.TRAIN_DIR,
              'yolov5/runs/detect'))
        target_name = f_name
        if start is not None and stop is not None:
            target_name = "/content/sourse.avi"
            ffmpeg_extract_subclip(f_name, start, stop, targetname=target_name)
        test_dict = dict()
        test_dict['weights'] = __class__.weights
        if save_labels:
            test_dict['save-txt'] = ''
        if save_crop:
            test_dict['save-crop'] = ''
        test_dict['conf'] = conf
        test_dict['source'] = target_name
        self.model.run(test_dict, exp_type='test')
