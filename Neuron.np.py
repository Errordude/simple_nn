#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 20:13:35 2020

@author: root
"""

import numpy as np

def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(neuron, weights, bias):
    neuron.weights = weights
    neuron.bias = bias

  def feedforward(neuron, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(neuron.weights, inputs) + neuron.bias
    return sigmoid(total)

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x))    # 0.9990889488055994
