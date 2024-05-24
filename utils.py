# make the necessary imports
import torch
from torchvision import datasets,models, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import os
from PIL import Image
import json
import numpy as np


# set batch_size
# batch_size = 64

# define a function to choose model architecture or pretrained model
def get_pretrained_model(architecture):
    try:
       model = torch.hub.load("pytorch/vision", architecture,pretrained=True )
    except RuntimeError as e:
       print(f"RunetimeError could not find model named {architecture} using efficientnet_b0 instead") 
       model = model = torch.hub.load("pytorch/vision", 'efficientnet_b0',pretrained=True)
    return model
# define function to build the classifier part of the network
def classifier(in_features:int,hidden_units:int):
        classifier = torch.nn.Sequential(
        OrderedDict([
            ('fc1',nn.Linear(in_features,hidden_units)),
            ('ReLU',nn.ReLU()),
            ('Dropout',nn.Dropout(0.2)),
            ('fc2',nn.Linear(hidden_units,hidden_units)),
            ('ReLU',nn.ReLU()),
            ('Dropout',nn.Dropout(0.2)),
            ('fc3',nn.Linear(hidden_units,102)),
            ('LogSoftmax',nn.LogSoftmax(dim=1))
        ])
        )
        return classifier
# defina a function to build the model 
def build_model(architecture,hidden_units):
    model = get_pretrained_model(architecture)
    # construct the classifier
    # freeze the pre trained model parameters
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model,'fc'):
        if isinstance(model.fc,nn.Linear):
            model.fc = classifier(model.fc.in_features,hidden_units)
        else:
            for i,layer in enumerate(model.fc):
                if isinstance(layer,nn.Linear):
                    model.fc = classifier(model.fc.in_features,hidden_units) 
                    break
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model,'classifier'):
        if isinstance(model.classifier,nn.Linear):
            model.classifier = classifier(model.fc.in_features,hidden_units)
        else:        
            for i,layer in enumerate(model.classifier):
                if isinstance(layer,nn.Linear):
                    model.classifier = classifier(model.classifier[i].in_features,hidden_units) 
                    break
        for param in model.classifier.parameters():
            param.requires_grad = True
    return model
    

# define a function to train the model
def train_loop(trainloader,validloader,model,loss_fn,optimizer,device,print_every=5):

    total_steps = len(trainloader)
    # set the model to train mode
 
    step = 0
   
    for images,labels in trainloader:
         train_loss = 0
         step +=1 
         images, labels = images.to(device), labels.to(device)
         # forward pass
         logps = model.forward(images)
         loss = loss_fn(logps,labels)
         
         # Backpropagation
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()
         train_loss += loss.item()
         if step%print_every == 0:
            model.eval()
            validation_loss = 0
            validation_accuracy = 0
            with torch.no_grad():
                for images,labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    # forward pass
                    logps = model.forward(images)
                    loss = loss_fn(logps,labels)
                    
                    # get the actual probabilities
                    ps = torch.exp(logps)
                    _ , top_class = ps.topk(1,dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    validation_loss += loss
                    validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
      
            print( f"Train loss: {train_loss/print_every:>7f}... "
                    f"Validation loss: {validation_loss/len(validloader):.3f}... "
                    f"Validation accuracy: {validation_accuracy/len(validloader):.3f}... "
                    f"step [{step:>5d}/{total_steps:>5d}]"
                    )
            model.train()
    # return the average loss for all the batches
    # return train_loss/print_every

           

def load_data(data_dir):
    train_dir = os.path.join(data_dir,'train')
    valid_dir = os.path.join(data_dir,'valid')
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]    
        )
    ])
    # The validation dataset should not contain the augmentations
    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]    
        )
    ])
    train_dataset = datasets.ImageFolder(train_dir,transform=train_transforms) 
    valid_dataset = datasets.ImageFolder(valid_dir,transform=valid_transforms) 
    trainloader = DataLoader(train_dataset,shuffle=True,batch_size=64)
    validloader = DataLoader(valid_dataset,shuffle=True,batch_size=64)
    return trainloader,validloader

def train(data_dir,print_every,hidden_units=512,architecture='efficientnet_b0',learn_rate=0.001,device='gpu',epochs=20):
    model = build_model(architecture,hidden_units)
    if device == 'cuda':
        device = ('cuda' if torch.cuda.is_available else 'cpu')
    # inialize loss and optimizer
    loss_fn = nn.NLLLoss()
    if hasattr(model,'fc'):
        optimizer = torch.optim.Adam(model.fc.parameters(),lr=learn_rate)
    elif hasattr(model,'classifier'):
        optimizer = torch.optim.Adam(model.classifier.parameters(),lr=learn_rate)
    model.to(device)
    train_dataloader, validation_dataloader = load_data(data_dir)
    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}\n-------------------------------")
        train_loss = train_loop(train_dataloader,validation_dataloader,model,loss_fn,optimizer,device=device,print_every=print_every)
        # validation_loss, validation_accuracy = validation_loop(validation_dataloader,model,loss_fn,device)
    # print the training loss, validation loss, and validation accuracy
    # return model, optimizer,train_dataloader.dataset.class_to_idx
    return model,architecture,hidden_units,optimizer,train_dataloader.dataset.class_to_idx
    

# defina a function to save the model
def save_model(save_dir,model,architecture,hidden_units,optimizer,class_to_idx):
   torch.save({
       'model_state_dict':model.state_dict(),
      'optimizer_state_dict':optimizer.state_dict(),
      'architecture':architecture,
      'hidden_units':hidden_units,
      'class_to_idx':class_to_idx
   }
       ,save_dir)


# load the model
def load_model(check_point_path):
    checkpoint= torch.load(check_point_path)
    architecture = checkpoint['architecture']
    hidden_units = checkpoint['hidden_units']
    class_to_idx = checkpoint['class_to_idx']
    model = build_model(architecture,hidden_units)
    return model,class_to_idx 

# load categories
def load_categories(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
# process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    pil_image = pil_image.resize((256,256))
    # set the crop size
    crop_size = 224
    # Get the details of the image
    width, height = pil_image.size
    # calculate the cropping box
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2 
    right = (width + crop_size) / 2
    bottom = ( height + crop_size ) / 2
    # crop the image
    pil_image = pil_image.crop((left,top,right,bottom))
    
    return pil_image


# Prediction
def predict(image_path, model,device, class_to_idx,cat_to_name,topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # load the image and resize
    image = process_image(image_path)
    model.to(device)
    # switch to evaluation mode
    model.eval()
    # convert pil image to numpy then to torch tensor and reshape to fit model input shape
    input_image = torch.from_numpy(np.array(image)).view(1,3,224,224)
    input_image = input_image.type(torch.FloatTensor).to(device)
    # pass the image to the model
    with torch.no_grad():
        logps = model.forward(input_image)
    # get the probabilities
    ps = torch.exp(logps)
    # get the top class and probs and return
    top_p,top_indices = ps.topk(topk,dim=1)
    # convert the classes to names
    # print(top_class[0])
    top_indices_numpy = top_indices.cpu().numpy()
    idx_to_class = {x: y for y, x in class_to_idx.items()}
    top_classes = [idx_to_class[x] for x in top_indices_numpy[0]]
    names = [cat_to_name[str(i)] for i in top_classes ]
    top_p_numpy = top_p.cpu().numpy()
    # plot the prediction of the image
    probabilities = top_p_numpy[0]
    for i, name in enumerate(names):
        print(f'{i+1}. prediction: {name} ......  probability: {probabilities[i]} ')
        
        
        