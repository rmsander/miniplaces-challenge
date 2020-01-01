 # 6.869 Miniplaces Challenge
 ## Overview
 This repository contains code for training and testing convolutional neural networks for image classification in PyTorch.  This was part of a class mini-project for MIT's 6.869 - Advances in Computer Vision.  We used a subset of the ImageNet dataset for this class containing 100 data labels.
 
 ## Environment Installation
 An Anaconda environment was used for this project.  To set up the conda environment, you can do so with the following bash command:
 
 `conda env create -f environment.yml`
 
 ## Training Options: Jupyter Notebooks and Python Files
 This framework is amenable for training and testing with both Jupyter notebooks and Python files, depending on your task (i.e. for `screen`/`tmux` sesssions with AWS EC2 machines, Python files should be used to avoid re-use or changing of variables with Jupyter notebooks).
 
 ## My Neural Network Submission
 Using AWS's EC2 (on an Ubuntu 18.04 Deep Learning AMI) with PyTorch, I trained a neural network with the following specifications to achieve a **71.7%** Top-5 Classification Accuracy (i.e. a prediction is considered correct if one of the top 5 predictions for a given image is the ground truth label):
 
 * **Architecture**: ResNet18
 * **Dropout Layers**: Penultimate Layer
 * **Optimizer**: ADAM
 * **Epochs**: 10
 * **Batch Size**: 128
 * **Learning Rate**: 0.001
 
## Credits
Thank you to the 6.869 team for sponsoring this mini project!

