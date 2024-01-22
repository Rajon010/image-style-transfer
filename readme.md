# Neural Image Style Transfer 

Exploring maximum-mean-discrepancy-based descriptive neural image style transfer using PyTorch.

## How to run

`./download_models.sh` provided by [9].  
`./download_images.sh`  
`python3 neural_image_style_transfer.py` built upon [9].

## Summary

This project studies the basic form of the neural image style transfer technique proposed by Gatys et al. ([1.1]). First, we reproduce the result of [1.1], which uses Gramian matrix to capture the concept of style. Next, we substitute mean vector, variance vector, and covariance matrix for Gramian matrix, and find that these new components capture different aspects of style and some of them even improve the quality of style transfer. Finally, we implement the idea "activation shift" in [3], which is another technique to improve the quality of style transfer.  
For details, refer to the [report](report.md).

## References

[1.1] L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. 2016.  
[3] R. Novak and Y. Nikulin. Improving the neural algorithm of artistic style. 2016.  
[9] PyTorch implementation of [1.1] by Gatys. https://github.com/leongatys/PytorchNeuralStyleTransfer
