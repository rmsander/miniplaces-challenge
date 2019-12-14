my_name = "Ryan Sander"
my_teammate = ""
my_team_name = "Nerdy Networks" # Your team name on the submission server

print("I'm %s. I worked with %s. I'm on team %s" % (my_name, my_teammate, my_team_name))

import os

# Root of data. Change this to match your directory structure. 
# Your submissions should NOT include the data.
# You might want to mount your google drive, if you're using google colab. 
# If you ran the cell above, your google drive will be located at '/content/gdrive/My Drive'
# datadir should contain train/ val/ and test/

# Sanity check #1: See our current directory, and make sure our data is present
#!pwd
#!ls

data_dir = os.path.join(os.getcwd(),"data/data/")
print("Data directory path is: {}".format(data_dir))

# List contents of data_dir for sanity check #2
print("Contents of data directory: {}".format(os.listdir(data_dir)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
# You might not have tqdm, which gives you nice progress bars
#!pip install tqdm
#from tqdm.notebook import tqdm
from tqdm import tqdm
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")

# Code for adding dropout layers
def add_dropout(M,p=0.5,num_ftrs=100):
    """Function that takes in a module M and outputs a new module with a 
    dropout layer appended in the fully connected region of the network."""
    M.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_ftrs))


# Code for initializing all the layers in a neural network
def init_weights(sub_mod, init=torch.nn.init.normal_):
    W = sub_mod.weight  # Get weights of layer
    init(W)

def initialize_model(model_name, num_classes, resume_from = None, \
                     resnet_model=models.resnet18, dropout=False, prob_drop=0.5):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # The model (nn.Module) to return
    model_ft = None
    # The input image is expected to be (input_size, input_size)
    input_size = 0
    
    # You may NOT use pretrained models!! 
    use_pretrained = False
    
    # By default, all parameters will be trained (useful when you're starting from scratch)
    # Within this function you can set .requires_grad = False for various parameters, if you
    # don't want to learn them

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = resnet_model(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224
        
    # Custom Models
    elif model_name == "fixresnet":
        from .resnext_wsl import resnext101_32x48d_wsl
        """FixResNeXt-101"""
        model=resnext101_32x48d_wsl()

    else:
        raise Exception("Invalid model name!")
    
    # If model isn't already in CPU, move it there
    model_ft = model_ft.to('cpu')
    if resume_from is not None:
        print("Loading weights from %s" % resume_from)
        model_ft.load_state_dict(torch.load(resume_from)['state_dict'])
    
    # Add dropout if desired
    if dropout:
        add_dropout(model_ft,p=prob_drop,num_ftrs=100)
        
    # Initialize the model
    #model_ft.apply(init_weights)
                     
    return model_ft, input_size

def get_dataloaders(input_size, batch_size, shuffle = True):
    # How to transform the image when you are loading them.
    # you'll likely want to mess with the transforms on the training set.
    
    # For now, we resize/crop the image to the correct input size for our network,
    # then convert it to a [C,H,W] tensor, then normalize it to values with a given mean/stdev. These normalization constants
    # are derived from aggregating lots of data and happen to produce better results.
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in data_transforms.keys()}
    # Create training and validation dataloaders
    # Never shuffle the test set
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False if x != 'train' else shuffle, num_workers=4) for x in data_transforms.keys()}
    return dataloaders_dict

# Create a helper function to plot losses
def plot_loss(epochs, train_losses, train_acc, val_losses, val_acc, model, save_dir, tag):
    
    # Create plot
    plt.plot(epochs, train_losses, color="r", label="Training Loss")
    plt.plot(epochs, val_losses, color="b", label="Validation Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses for {} as a Function of Epochs".format(model))
    
    # Make output directory if it doesn't exist already
    out_dir = os.path.join(os.getcwd(),"results/",tag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Save model to appropriate output directory
    plt.savefig(os.path.join(os.getcwd(),save_dir,"losses.png"))
    
    # Make sure the figure is closed before proceeding
    plt.clf()


def make_optimizer(model, opt=torch.optim.SGD, lr=0.001):
    # Get all the parameters
    params_to_update = model.parameters()
    print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    # Use SGD
    #optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    # Make custom optimizer
    optimizer = opt(params_to_update, lr=lr)
    return optimizer

def get_loss():
    # Create an instance of the loss function
    criterion = nn.CrossEntropyLoss()
    return criterion

def train_model(model, dataloaders, criterion, optimizer, save_dir = None, save_all_epochs=False, \
                num_epochs=25, tag=None, string_graph_name="ResNet18"):
    '''
    model: The NN to train
    dataloaders: A dictionary containing at least the keys 
                 'train','val' that maps to Pytorch data loaders for the dataset
    criterion: The Loss function
    optimizer: The algorithm to update weights 
               (Variations on gradient descent)
    num_epochs: How many epochs to train for
    save_dir: Where to save the best model weights that are found, 
              as they are found. Will save to save_dir/weights_best.pt
              Using None will not write anything to disk
    save_all_epochs: Whether to save weights for ALL epochs, not just the best
                     validation error epoch. Will save to save_dir/weights_e{#}.pt
    '''
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # print the "tag" that we use to compare and separate different networks
    print("Tag for this training session: {}".format(tag))
    
    # Store epochs and losses so we can plot them
    epochs = [i+1 for i in range(num_epochs)]
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # TQDM has nice progress bars
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # torch.max outputs the maximum value, and its index
                    # Since the input is batched, we take the max along axis 1
                    # (the meaningful outputs)
                    _, preds = torch.max(outputs, 1)

                    # backprop + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Append training loss and accuracy
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)
		# Save the model after the epoch
                checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(save_dir,'checkpoint.pth.tar'))
            
            # Append validation loss and accuracy
            elif phase == 'val':
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            
            
        print()
    
    # Plot losses
    plot_loss(epochs, train_losses, train_acc, val_losses, val_acc, string_graph_name, save_dir, tag)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet]
# You can add your own, or modify these however you wish!
model_name = "resnet"

# Number of classes in the dataset
# Miniplaces has 100
num_classes = 100

# Batch size for training (change depending on how much memory you have)
# You should use a power of 2.
batch_size = 4

# Shuffle the input data?
shuffle_datasets = True

# Number of epochs to train for 
num_epochs = 2

# Save weights for all epochs, not just the best one
save_all_epochs = True

# Other custom hyperparameters
resnet_models=[models.resnet101]  # CHANGE
string_graph_names = ["ResNet101"]
dropouts=[False]
prob_drops=[0.5, 0.5, 0.5, 0.5] 
opts=[torch.optim.SGD]  ## CHANGE
tags = ["resnet101_dropout_False_opt_SGD_lr_{}"]

for resnet_model, string_graph_name, dropout, prob_drop, opt, tag in zip(resnet_models, string_graph_names, dropouts, \
                                                                        prob_drops, opts, tags):
    for lr in [0.01]:
        it_tag = tag.format(lr)  # Create tag with learning rate

        ### IO
        # Path to a model file to use to start weights at
        resume_path = os.path.join(os.getcwd(),"models/", it_tag)
        if not os.path.exists(resume_path):
            os.mkdir(resume_path)
    
        print("Number of files in resuming path: {}".format(len(os.listdir(resume_path))))
        print("Learning rate is: {}".format(lr))
        print("Tag is: {}".format(it_tag))
    
        if len(os.listdir(resume_path)) <= 1:  # If no models are saved
            resume_from = None
    
        else:  # If we have saved models
            resume_from = os.path.join(resume_path,"checkpoint.pth.tar")

        # Directory to save weights to
        save_dir = resume_path
        print("Save directory: {}".format(save_dir))
        os.makedirs(save_dir, exist_ok=True)

        # Initialize the model for this run
        model, input_size = initialize_model(model_name = model_name, num_classes = num_classes, resume_from = resume_from, \
                                         resnet_model=resnet_model, dropout=dropout, prob_drop=prob_drop)
        dataloaders = get_dataloaders(input_size, batch_size, shuffle_datasets)
        criterion = get_loss()
    
        # Move the model to the gpu if needed
        model = model.to(device)

        optimizer = make_optimizer(model, opt=opt, lr=lr)

        # Train the model!
        trained_model, validation_history = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
                   save_dir=save_dir, save_all_epochs=save_all_epochs, num_epochs=num_epochs, tag=it_tag, string_graph_name=string_graph_name)
