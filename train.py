# make the necessary imports
import torch
from torchvision import datasets,models, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import os
import argparse

# set batch_size
batch_size = 64

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

def validation_loop(dataloader,model,loss_fn,device):
    # set the model to train mode
    step = 0
    model.eval()
    validation_loss = 0
    validation_accuracy = 0
    with torch.no_grad():
        for images,labels in dataloader:
            step += 1
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
    return validation_loss/len(dataloader),validation_accuracy/len(dataloader)
           

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
    return model
    

# defina a function to save the model
def save_model(save_dir,model):
   torch.save(model,save_dir)
   
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Trainer",
                                     description="My Custom tool for training computer vision models from pretrained models")
    parser.add_argument("data_dir", help="Path to the dataset")
    parser.add_argument("--arch", default='efficientnet_b0',help="The pretrained model you want to use for training")
    parser.add_argument("--hidden_units", type=int,default=256,help="The number of hidden units for the classifier model")
    parser.add_argument("--epochs", type=int,default=20,help="The number of hidden units for the classifier model")
    parser.add_argument("--learning_rate", type=float,default=0.001,help="The pretrained model you want to use for training")
    parser.add_argument("--save_dir",help="The path where your model would be saved after training")
    parser.add_argument("--print_every",type=int,default=2,help="The path the amount")
    parser.add_argument("--gpu",action='store_true',help="Run this option if you want to train on gpu")
    # Parse arguments
    args = parser.parse_args()
    device = 'cpu'
    if args.gpu:
        device ='cuda'
    
        
    model=train(args.data_dir,args.print_every,hidden_units=args.hidden_units,architecture=args.arch,
          learn_rate=args.learning_rate,epochs=args.epochs,device=device)

    if args.save_dir:
        save_model(args.save_dir,model)
    
    