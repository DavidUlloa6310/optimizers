# Optim: Optimization Algorithms for Deep Learning

A simple package for implementing and benchmarking popular optimization algorithms used in deep learning.

## Overview
This repository provides implementations of common optimization algorithms used in machine learning, with a focus on comparing their performance across different datasets and model architectures. The package uses JAX and Flax for efficient, hardware-accelerated implementations.

## Installation

Install the package directly from the repository, if on TPU:

```bash
pip install -e .[tpu]
```

or with developer packages:

```bash
pip install -e .[tpu, dev] # installing both - but can just do [dev] if needs be
```

## Implemented Optimizers

The package currently implements the following optimization algorithms:

1. **Stochastic Gradient Descent (SGD)**: Basic implementation with fixed learning rate
2. **SGD with Momentum**: Introduces velocity term to accelerate training
3. **SGD with Nesterov Momentum**: Calculates gradients after applying the velocity
4. **AdaGrad**: Adapts learning rates based on parameter history
5. **RMSProp**: Extends AdaGrad with exponential moving average
6. **Adam**: Combines momentum and adaptive learning rates

## Benchmarking

The package includes benchmarking functionality for evaluating optimizer performance on standard datasets:

**Supported Datasets**

* MNIST: Handwritten digit classification
* CIFAR-10: Image classification
* IMDB: Sentiment analysis

### Running Benchmarks

```bash
python -m optim.benchmarks.bench --graph-directory ./results --mnist --cifar --imdb
```

### Arguments:

* `--graph-directory`: Directory to save performance plots
* `--mnist`: Run benchmarks on MNIST dataset
* `--cifar`: Run benchmarks on CIFAR-10 dataset
* `--imdb`: Run benchmarks on IMDB dataset

### Visualization
For each optimizer and dataset combination, the benchmarking tool generates plots showing:

* Loss curve
* Accuracy curve
* Gradient norm

## References

The optimization algorithms implemented in this package are based on the following papers:

Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7).

Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 4(2), 26-31.

Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. In International conference on machine learning (pp. 1139-1147).

Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence O(1/k^2). Doklady ANSSSR, 269, 543-547.
