The training is computational intensive and you may need to run this on GPU or alternativly you may run it on CoLab

1) install all require packages like tensorflow
2) open the terminal and go to the directory 
3) run model training script 'tf_fine_tune.py' by typing: python tf_fine_tune.py
4) wait until it's done, the model.h5 will then be saved
5) run model validation script 'tf_deploy_02.py' by typing: python tf_deploy_02.py
6) if you want to do single image classification, run script 'tf_deploy.py' by typing: python tf_deploy.py  