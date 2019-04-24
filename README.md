# Statistical Guarantees for Robustness of Bayesian Neural Networks

#### Code for paper:

https://arxiv.org/abs/1903.01980

-------------------------------------

In this repository you will find all of the code and data to run and replicate the
results of our paper on Statistical Guarantees for Robustness of Bayesian Neural Networks.
This includes, all the code to train deep feed forward and deep convolutional
neural nets in a Bayesian fashion with Edward and Tensorflow. 

Prior to running the code in this repository we recommend you run the following
command in order to ensure that all of the correct dependencies are used:

```
$ virtualenv -p /usr/bin/python2.7 virt
$ source virt/bin/activate
$ pip install -r requirements.txt
```

This will activate the virtual environment for the project.

From there, you can go into either the MNIST, MNIST-veri, or GTSRB directories for either
of those two datasets. In each respective directory there are sub directories
for Monte Carlo Dropout (MCDropout), Hamiltonian Monte Carlo (HMC), and Variational
Inference (VI) training of the architectures reported in the paper.

Robustness tests of these networks was done with OpenCV and SciKit image
for invariance tests and then with the cleverhans library for known attacks,
and the DeepGO algorithm (in MATLAB) for verification/reachability.

-------------------------------------

Note: Since the publication of this paper and running of the results Tensorflow 2.0.0 
has come out. In the coming month we will update all of the code to the latest versions
of Tensorflow, Cleverhans, and Edward.

