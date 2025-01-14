
# Weight matrices compression based on PDB model in deep neural networks


### Abstract
Recently, weight matrix compression has been demonstrated to effectively reduce overfitting and improve the generalization performance of deep neural networks. Compression is primarily achieved by filtering out noisy eigenvalues of the weight matrix.
In this work, a novel Double Bulk (DB) model is proposed to characterize the eigenvalue behavior of the weight matrix, which is more suitable than the existing Unit Bulk (UB) model. Based on this model and Random Matrix Theory (RMT), we have discovered a new method for determining the boundary between noisy eigenvalues and information. We have also introduced a noise-filtering algorithm to reduce the rank of the weight matrix and adjust its dominant eigenvalues. Experiments show that our DB model fits the empirical distribution of eigenvalues of the weight matrix better than the UB model, and our compressed weight matrices have lower rank at the same level of test accuracy. In some cases, our compression method can even improve generalization performance when labels contain noise.

### Description
- model: store the trained model and parmaters
- estimation.py: Double bulk least squares algorithm to estimate the initial number of spikes $k_0$, the proportion of $\sigma_1^2$ among all bulk eigenvalues $t$, $\sigma_1^2$ and $\sigma_2^2$
- FCNN_MNIST.py: train fully connected neural network on MNIST
- neural.py: load and process neural networks
- noise_filter.py: to evaluate the compressed model aftering recover and remove singular values
- figure.ipynb: generate the figures in the paper


### Requirements
The network model in this project was generated using tensorflow.keras. However, due to disk space limitations, we cannot upload all the trained networks and the precomputations to github. You can run the training scripts to reproduce the experiments.

