import json
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision
from torch import nn, optim
from collections import OrderedDict
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable



def load_data(root = "ImageClassifier/flowers"):
    
    with open('ImageClassifier/cat_to_name.json', 'r') as f:      
        cat_to_name = json.load(f)
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
        
    data_transforms_training = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])

    data_transforms_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    data_transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    train_set = datasets.ImageFolder(
    train_dir,
    transform = data_transforms_training
    )

    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    valid_set =  datasets.ImageFolder(
        valid_dir,
        transform = data_transforms_valid
    )


    test_set =  datasets.ImageFolder(
        test_dir,
        transform = data_transforms_test
    )


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders = 
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = 64)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
    
    return train_loader, valid_loader, test_loader, train_set

def network_construct(structure='vgg19',hidden_units=1000, lr=0.001, device='gpu'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    struct_dict = {'vgg19': 25088,
                   'densenet121': 1024,
                    'alexnet':9216}
    
    if structure == 'vgg19':
        model = models.vgg16(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model.Did you mean vgg19,densenet121,or alexnet?".format(structure))
    
    for param in model.parameters():
        param.requires_grad = False

        #create a network for classification        
        model.classifier = nn.Sequential(nn.Linear(struct_dict[structure], hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units, 256),
                                         nn.ReLU(),
                                         nn.Dropout(0.3),
                                         nn.Linear(256,102),
                                         nn.LogSoftmax(dim=1))
        
        #displaying model architecture
    model
        # activate neural network computations in available device
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    if torch.cuda.is_available() and device == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    return model, criterion

def save_checkpoint(train_set, model = 0, save_dir = 'checkpoint.pth', structure = 'vgg19', hidden_units = 1000, lr = 0.001, epochs = 1):       
    model.class_to_idx =  train_set.class_to_idx
    torch.save({'structure' :structure,
                'hidden_units':hidden_units,
                'learning_rate':lr,
                'epochs':epochs,
                'model_state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                save_dir)
def load_checkpoint(save_dir = 'checkpoint.pth'):
    checkpoint = torch.load(save_dir)
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    epochs = checkpoint['epochs']
    structure = checkpoint['structure']

    model, _ = network_construct(structure, hidden_units, lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model  

def predict(image_path, model, topk=5, device='gpu'):   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    
    np_img = process_image(image_path)
    np_img.unsqueeze_(0)
    probabs = torch.exp(model.forward(np_img.to(device)))
    top_probabs, top_labels = probabs.topk(topk)
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key
    np_top_labels = top_labels[0].cpu().numpy()

    top_labels = [int(idx_to_class[label]) for label in np_top_labels]
    
    return top_probabs, top_labels
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)

    return image

        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        