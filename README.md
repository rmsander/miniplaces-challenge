 # 6.869 Miniplaces Challenge
 ## Overview
 This repository contains code for training and testing convolutional neural networks for image classification in PyTorch.  This was part of a class mini-project for MIT's 6.869 - Advances in Computer Vision.  We used a subset of the ImageNet dataset for this class containing 100 data labels.
 
 ## Environment Installation
 An Anaconda environment was used for this project.  To set up the conda environment, you can do so with the following bash command:
 
 `conda env create -f environment.yml`
 
 ## Port Forwarding for Remote Jupyter Notebooks
 To use Jupyter notebooks on a remote host (e.g. AWS EC2), see the bash script `AWS_ssh.sh` and replace the `ami_key.pem` and `remote_user@remote_host` placeholders with your relevant remote host key (if applicable) and remote machine, respectively.  This script forwards remote Jupyter notebooks from the 9999 port (default) to the 8888 port (in case 9999 is being used for a local Jupyter notebook).  
 
 If you prefer, you can run the `AWS_ssh.sh` bash script to ssh into your remote host and set the remote port for Jupyter notebooks to 8888.  Once you have ssh'ed into your remote machine, you can type the following command (note: you must have Jupyter notebook already installed) to start a Jupyter notebook that can be accessed from your local browser:
 
 `jupyter notebook --no-browser`
 
 After typing this, you should see a URL for your Jupyter notebook in the prompt.  Copy and paste this URL into your local browser, and replace the port number `9999` with `8888`.  This should enable you to access your remote Jupyter notebook session from your local browser.
 
 
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
 
 ## Reports
 You can find my reports for this project in this repository:
 
 * [**report_I.pdf**](https://github.com/rmsander/miniplaces-challenge/blob/master/report_I.pdf) (Part I)
 * [**report_II.pdf**](https://github.com/rmsander/miniplaces-challenge/blob/master/report_II.pdf) (Part II)
 
## Credits
Thank you to the 6.869 team for sponsoring this mini project!

