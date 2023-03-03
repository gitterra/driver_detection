import os
import re
from . import bcolors

class Worker:
    def __init__(self):
        self.index_class = 0

    def analysis(self, directory):
        print(f'{bcolors.OKBLUE}Этап: {bcolors.BOLD}НЕПРИСТЕГНУТЫЙ РЕМЕНЬ БЕЗОПАСНОСТИ', end='')
        try:
            result = []
            if os.path.isdir(directory):
                for f in sorted(os.listdir(directory)):
                    with open(f'{directory}/{f}', 'r') as fl:
                        data = fl.read().split()
                    belt_exist = False
                    for i in range(0, len(data), 5):
                        if int(data[i]) == self.index_class:
                            belt_exist = True
                            break
                    if not belt_exist:
                        result.append(int(re.findall('[0-9]+', f)[0]))
            print(f'{bcolors.ENDC}{bcolors.OKGREEN} Done{bcolors.ENDC}')
            return sorted(result)
        except:
            print(f'{bcolors.ENDC}{bcolors.OKGREEN} Done{bcolors.ENDC}')
            return []
    
    
