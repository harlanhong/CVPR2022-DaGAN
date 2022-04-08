import os
import random
import csv
import pdb
import numpy as np

def create_csv(path):
    videos = os.listdir(path)
    source = videos.copy()
    driving = videos.copy()
    random.shuffle(source)
    random.shuffle(driving)
    source = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    zeros = np.zeros((len(source),1))
    content = np.concatenate((source,driving,zeros),1)
    f = open('vox256.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    csv_writer.writerows(content)
    f.close()
    

if __name__ == '__main__':
    create_csv('/data/fhongac/origDataset/vox1/test')