# ResNet34
This repository contains my implementation of the paper "Deep Residual Learning for Image Recognition," one of the first papers in which a neural network with hundreds of layers was trained successfully. In my implementation, I build each module directly from tensors in PyTorch, rather than relying on pre-programed modules like PyTorch's Conv2d and BatchNorm2D.

Using the pretrained weights of torchvision's ResNet34 model, my model achieves performance similar to that of torchvision's ResNet34. Trained from scratch on a smaller dataset, my model achieves a test accuracy of approximately 71% after eight training epochs.

This implementation was guided by Redwood Research's MLAB AI safety bootcamp curriculum. Many thanks to Redwood Research for allowing my university to run a seminar using the MLAB curriculum.
