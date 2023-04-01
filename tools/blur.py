import os
import re
from . import bcolors

class Worker:
    def __init__(self):
        self.index_class = 3

    def analysis(self, directory):
        print(f'{bcolors.OKBLUE}Этап: {bcolors.BOLD}РАЗМЫТОЕ ИЗОБРАЖЕНИЕ', end='')
        try:
          result = []            
          if os.path.isdir(directory):            
              filenames = os.listdir(directory)
              numbers = [int(re.findall('\d+', filename)[0]) for filename in filenames]

              all_numbers = set(range(min(numbers), max(numbers) + 1))
              result = sorted(list(all_numbers - set(numbers)))
          print(f'{bcolors.ENDC}{bcolors.OKGREEN} Done{bcolors.ENDC}')
          return sorted(result)
        except:
            print(f'{bcolors.ENDC}{bcolors.FAIL} Done{bcolors.ENDC}')
            return []
