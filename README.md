# AH-Net
The Pytorch implementation of the 3D Anisotropic Hybrid Network (AH-Net) that transfers convolutional features learned from 2D images to 3D anisotropic volumes. Such a transfer inherits the desired strong generalization capability for within-slice information while naturally exploiting between-slice information for more effective modelling. We experiment with the proposed 3D AH-Net on two different medical image analysis tasks, namely lesion detection from a Digital Breast Tomosynthesis volume, and liver and liver tumor segmentation from a Computed Tomography volume and obtain the state-of-the-art results.

For more details, please refer to the paper:
Siqi Liu, Daguang Xu, S. Kevin Zhou, Thomas Mertelmeier, Julia Wicklein, Anna Jerebko, Sasa Grbic, Olivier Pauly, Weidong Cai, Dorin Comaniciu
3D Anisotropic Hybrid Network: Transferring Convolutional Features from 2D Images to 3D Anisotropic Volumes
arXiv:1711.08580 [cs.CV]


We only host the network modules here for brevity. 
To train a model for 3D medical images:
1. Pretrain the 2D model from `net2d.FCN` or `net2d.MCFCN`. This 2D FCN model is initialised with the Pytorch officially released ResNet50.
2. Copy the trained 2D model to the 3D AH-Net as 
```
net = AHNet(num_classes=2)
net.copy_from(model2d)
```
3. Train the AH-Net model as a 3D fully convolutional network
