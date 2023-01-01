

import argparse
from torch.autograd import Variable
import torch
from torch import nn, optim
import torchvision.datasets as datasets
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import json
from PIL import Image
import utils

parser = argparse.ArgumentParser(description = 'prediction parser')

parser.add_argument('input', default='ImageClassifier/flowers/test/15/image_06374.jpg', nargs='?', action="store", type = str)
parser.add_argument('--gpu', default="gpu", action="store", help="select engine (cpu|gpu)",dest="gpu")
# parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('--top_k', default=5, dest="top_k",type=int,help="input number of topk",action="store", )
parser.add_argument('--category_names', dest="category_names", action="store", default='ImageClassifier/cat_to_name.json')
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)


args = parser.parse_args()

device = args.gpu
names = args.category_names
topk_outputs = args.top_k
image_path = args.input
save_path = args.checkpoint

def final_predict():
    model=utils.load_checkpoint(save_path)
    with open(names, 'r') as json_f:
        name = json.load(json_f)
    
    top_probabs, top_labels = utils.predict(image_path, model, topk_outputs, device) 
    top_probabs = top_probabs[0].cpu().detach().numpy() 
    top_flowers = [name[str(lab)] for lab in top_labels]
    
    i = 0
    while i < topk_outputs:
        print("{} with a probability of {}".format(top_flowers[i], round(top_probabs[i],3) ))
        i += 1
    print("")

    
if __name__== "__main__":
    final_predict()



