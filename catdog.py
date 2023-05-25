import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from torchvision.io import read_image
import os
from PIL import Image

class catDogDataset(torch.utils.data.Dataset):
    def __init__(self, rootFolder):
        self.rootFolder = rootFolder
        self.paths = []
        self.labels = []
        self.getImgsAndLabels()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,  index):
        image = self.paths[index]
        return self.reformat(image), self.labels[index]

    def reformat(self, img_path):
        '''This function take in a image path and return it as a (1, 28, 28) tensor to pass into the NN Model
        
        Attribute:
        ---------
        img_path : the path to the image on the computer
        '''
        img = np.asarray(Image.open(img_path)) 

        #resize to 28x28
        img = cv.resize(img, (148, 148)) 

        if (img.shape != (148, 148)):
            #convert to gray scale
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #note that color is still from 0 to 255


        #Making the color to be between 0 and 255
        img = np.float32(img / 255)

        #reshape and convert to tensor
        img_ = torch.from_numpy(img).reshape(1, 148, 148)
        return img_

    
    def getImgsAndLabels(self):
        #get the list of class folders
        # os.listdir(self.rootFolder) lists all the items (files and folders) in the self.rootFolder directory.
        #os.path.isdir(os.path.join(self.rootFolder, folder_name) return a boolean value to indicate if the path is a directory
        class_folders = [folder_name for folder_name in os.listdir(self.rootFolder) if os.path.isdir(os.path.join(self.rootFolder, folder_name))]

        #mapping the img path with their labels
        for class_folder in class_folders:
            #Extracting the class labels and img file names
            if (class_folder == "Cat"):
                class_label = 1
            else:
                class_label = 0

            class_folder_path = os.path.join(self.rootFolder, class_folder) #join automatically assign the correct separator between the folders names
            #if file name ends with 'jpg' then add to list
            img_names = [img_name for img_name in os.listdir(class_folder_path) if img_name.endswith('.jpg')]
            
            #Getting the complete path directories for  the images and append it to the self.image_paths list
            self.paths.extend([os.path.join(class_folder_path, img_name) for img_name in img_names])
            #Getting the labels
            self.labels.extend([class_label] * len(img_names))


train_set = catDogDataset(r"C:/pytorch/Output/train")
test_set = catDogDataset(r"C:/pytorch/Output/test")
val_set = catDogDataset(r"C:/pytorch/Output/val")

train_dataLoader = DataLoader(dataset = train_set,
                        batch_size = 70, 
                        shuffle= True,
                        drop_last= True)
for picture, label in train_dataLoader:
    print(f"Shape of X [N, C, H, W]: {picture.shape}")
    print(f"Shape of y: {label.shape} {label.dtype}")
    break
len(train_dataLoader)

test_dataLoader = DataLoader(dataset = test_set,
                             batch_size = 70,
                             shuffle= True,
                             drop_last= True)
val_dataLoader = DataLoader(dataset = val_set,
                            batch_size = 70,
                            shuffle= True,
                            drop_last= True)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class CatDogModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(148*148, 300), #input to first layer
            nn.ReLU(),
            nn.Linear(300, 200), #first to second layer
            nn.ReLU(),
            nn.Linear(200, 100), #second to third layer
            nn.ReLU(),
            nn.Linear(100, 2) #third layer to output cat or dog
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = CatDogModel().to(device)

loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fun, optimizer):
    model.train()

    for batch, (pic, lab) in enumerate(dataloader):
        pic, lab = pic.to(device), lab.to(device)

        # Compute prediction error
        pred = model(pic)
        loss = loss_fun(pred, lab)

        # Backpropagation
        loss.backward() #compute the loss function's gradients
        optimizer.step() #update the parameters
        optimizer.zero_grad() #clear the model's gradient to avoid gradient accumulation

def test_validate(dataloader, model, loss_fun):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for pic, lab in dataloader:
            pic, lab = pic.to(device), lab.to(device)
            pred = model(pic)
            test_loss += loss_fun(pred, lab).item() #get the loss
            correct += (pred.argmax(1) == lab).type(torch.float).sum().item() #get how many times the model guess correctuly
    test_loss /= num_batches #loss per batch
    correct /= size #accuracy
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5

for i in range(epochs):
    print("epoch:", i + 1)
    train(train_dataLoader, model, loss_fun, optimizer)
    test_validate(val_dataLoader, model, loss_fun)
print("done lol")