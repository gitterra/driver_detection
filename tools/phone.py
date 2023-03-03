import os
import re


class Worker:
    def __init__(self):
        self.index_class = 4

    def analysis(self, directory):
        print(f'{bcolors.OKBLUE}Этап: {bcolors.BOLD}ИСПОЛЬЗОВАНИЕ ТЕЛЕФОНА', end='')
        try:
            result = []
            if os.path.isdir(directory):
                for f in sorted(os.listdir(directory)):
                    with open(f'{directory}/{f}', 'r') as fl:
                        data = fl.read().split()
                    for i in range(0, len(data), 5):
                        if int(data[i]) == self.index_class:
                            result.append(int(re.findall('[0-9]+', f)[0]))
                            break
            print(f'{bcolors.ENDC}{bcolors.OKGREEN} Done{bcolors.ENDC}')
            return sorted(result)
        except:
            print(f'{bcolors.ENDC}{bcolors.FAIL} Done{bcolors.ENDC}')
            return []
