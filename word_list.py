import data_original
import re

class generate_word_list(object):
    def __init__(self, path):
        Data = data_original.JSONDataset(path)
        self.all_data = Data.items
        self.length = len(self.all_data)
        self.gloss_list=[]
        self.word_list=[]
        self.init()

    def init(self):
        for item in range(self.length):
            gloss = self.all_data[item]['gloss']
            self.gloss_list.append(gloss)

        for item in range(self.length):
            a = self.gloss_list[item]
            # a = a.lower()
            # pattern = r'\,'
            # a = re.sub(pattern, ' ', a)
            list = re.split(r'\s+', a)
            # if a[-1] == '.':
            #     list.remove('.')
            self.word_list.append(list)
