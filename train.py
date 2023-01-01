print('hello world')
# %config InlineBackend.figure_format = 'retina'
import argparse 
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plts
import torch.nn as nn
from torch import optim
from PIL import Image
import numpy as np
import torch.nn.functional as F
import utils


parser = argparse.ArgumentParser(description='collects variables')
parser.add_argument('data_dir', nargs='*', action="store", default="ImageClassifier/flowers")
parser.add_argument("--savedir", action="store", type=str, default='./checkpoint.pth',help="input save directory")
parser.add_argument("--arch", action="store", type=str, default="vgg19", help = "input architecture")
parser.add_argument("--learn_rate", action="store", type=float, default=0.001, help = "input learning rate")
parser.add_argument("--hidden_units", action="store", dest="hidden_units", type=int, default=512, help = "input hidden units amount")
parser.add_argument("--epochs", action="store", type=int, default=3, help = "epochs quantity")        
parser.add_argument("--gpu", action="store", default="gpu", help = "enter gpu")


args = parser.parse_args()
data_directory = args.data_dir
save_dir = args.savedir
lr = args.learn_rate
structure = args.arch
hidden_units = args.hidden_units
engine = args.gpu
epochs = args.epochs



if torch.cuda.is_available() and engine == 'gpu':
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def train():
    train_loader, valid_loader, test_loader, train_set = utils.load_data(data_directory)
    model, criterion = utils.network_construct(structure,hidden_units,lr,engine)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    
    steps=0
    running_loss = 0
    print_interval = 5
    
    print('training is starting')
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            steps += 1
            if torch.cuda.is_available() and engine =='gpu':
                images,labels = images.to(device), labels.to(device)
                model = model.to(device)
                optimizer.zero_grad()   

                #foward propagate input images through network and get loss
                logits = model.forward(images)


                loss = criterion(logits, labels)  
                running_loss += loss.item()

                #Backward propagate loss through netwotk
                loss.backward()

                #Update weights in a direction that reduces the loss
                optimizer.step()

                #TEST MODEL ON SEPERATE DATA
                if steps % print_interval == 0:
                    model.eval()

                    test_loss = 0
                    accuracy = 0

                    with torch.no_grad():
                        for images, labels in test_loader:
                            images, labels = images.to(device), labels.to(device)
                            #forward images through neuralnet & calc loss
                            logits = model.forward(images)
                            loss_2 = criterion(logits, labels)
                            test_loss += loss_2.item()

                            #Get accuracy of prediction
                            ps = torch.exp(logits)
                            top_p, top_class = ps.topk(1,dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()     

                            print(f'Epoch {epoch+1}/{epochs}..'
                                  f'Train loss: {running_loss/print_interval:.3f}..'
                                  f'Test loss: {test_loss/len(test_loader):.3f}..'
                                  f'Test accuracy: {accuracy/len(test_loader) * 100:.2f}%')
                            
                    running_loss = 0
                    model.train()
    model.class_to_idx =  train_set.class_to_idx
    torch.save({'structure' :structure,
                'hidden_units':hidden_units,
                'learning_rate':lr,
                'epochs':epochs,
                'model_state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                save_dir)
    print("Saved checkpoint!")                    

if __name__ == "__main__":
    train()   
  
     
    
    