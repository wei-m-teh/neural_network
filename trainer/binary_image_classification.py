'''
Deep Neural Network for Image Classification: Application.

Binary Classification problem. Identifies whether a given image is a cat or non cat.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from trainer.dnn_app_utils_v3 import *

def load_datasets():
    '''
    Loads data and apply some transformations, such as reshapes, standardizations to
    make datasets structured appropriately for model training.

    Original Image dimensions are 64 x 64 x 3.

    :return:
    '''
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    train_x_flatten = train_x_orig.reshape((train_x_orig.shape[0], 64*64*3)).T
    test_x_flatten = test_x_orig.reshape((test_x_orig.shape[0], 64*64*3)).T
    train_x = train_x_flatten / 255 # Standardize the image pixels
    test_x = test_x_flatten / 255 # Standardize the image pixels
    return train_x, train_y, test_x, test_y, classes

def init_params(network_dim):
    l_idx = 0
    parameters = {}
    for l in network_dim:
        if l_idx == 0:
            l_idx += 1
            continue
        else:
            param_w_name = "{}{}".format("W", l_idx)
            param_b_name = "{}{}".format("B", l_idx)
            parameters[param_w_name] = np.random.randn(network_dim[l_idx], network_dim[l_idx - 1]) * 0.01
            parameters[param_b_name] = np.zeros((network_dim[l_idx],1))
        l_idx += 1
    return parameters

def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    #cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)
    return A

def sigmoid(Z):
    s = 1 / ( 1 + np.exp(-Z))
    return s

def feed_forward(x, y, network_dim, params, layer_act_func, final_act_func):
    layers = len(network_dim) - 1
    zs = {}
    acts = {}
    placeholder = x
    for l in range(1,layers+1):
        z_name = "Z{}".format(l)
        act_name = "A{}".format(l)
        zs[z_name] = params["W{}".format(l)].dot(placeholder) + params["B{}".format(l)]
        if l < layers:
            if layer_act_func == "relu":
                acts[act_name] = relu(zs[z_name])
            elif layer_act_func == "sigmoid":
                acts[act_name] = sigmoid(zs[z_name])
            else:
                raise Exception("Unsupported activation function")
        else:
            if final_act_func == "sigmoid":
                acts[act_name] = sigmoid(zs[z_name])
            elif final_act_func == "relu":
                acts[act_name] = relu(zs[z_name])
            else:
                raise Exception("unsupported activation function")
        placeholder = acts[act_name]
    return zs, acts

def cost_function(y_head, y):
    total_loss = -y.dot(np.log(y_head).T) - (1 - y).dot(np.log(1 - y_head).T)
    cost = total_loss / y.shape[1]
    return np.squeeze(cost)

def backward_propagation(params, zs, acts, x, y, network_dims, layer_act_func, final_act_func):
    layers = len(network_dims) - 1
    examples = x.shape[1]
    d = {}
    for l in range(layers, 0, -1): # calculate derivative backwards
      dz_name = "dZ{}".format(l)
      dw_name = "dW{}".format(l)
      db_name = "dB{}".format(l)
      if l == layers:
          d[dz_name] = acts["A{}".format(l)] - y
      else:
          if layer_act_func == "relu":
              def g_prime(Z):
                  dZ = Z > 0
                  return dZ.astype(np.float32)
              dz = params["W{}".format(l+1)].T.dot(d["dZ{}".format(l+1)]) * g_prime(zs["Z{}".format(l)])
              d[dz_name] = dz
          else:
              raise Exception("Unsupported activation function!")

      if l == 1:
        d[dw_name] = d[dz_name].dot(x.T) / examples
      else:
        d[dw_name] = d[dz_name].dot(acts["A{}".format(l - 1)].T) / examples

      d[db_name] = np.sum(d[dz_name], axis=1, keepdims=True)
    return d

def update_params(d, params, learning_rate, network_dims):
    layers = len(network_dims) - 1
    for l in range(1, layers+1):
        orig_w = params["W{}".format(l)]
        orig_b = params["B{}".format(l)]
        #print("orig W{0}: {1}, orig B{0}: {2}".format(l, orig_w, orig_b))
        params["W{}".format(l)] = params["W{}".format(l)] - learning_rate * d["dW{}".format(l)]
        params["B{}".format(l)] = params["B{}".format(l)] - learning_rate * d["dB{}".format(l)]
        new_w = params["W{}".format(l)]
        new_b = params["B{}".format(l)]
        #print("updated W{0}: {1}, updated B{0}: {2}".format(l, new_w, new_b))
    return

def main():
    train_x, train_y, test_x, test_y, classes = load_datasets()
    network_dims = (train_x.shape[0], 7, 1)
    params = init_params(network_dims)
    iterations = 1000
    layer_act_func = "relu"
    final_act_func = "sigmoid"
    learning_rate = 0.01
    for i in range(iterations):
        zs, acts = feed_forward(train_x, train_y, network_dims, params, layer_act_func, final_act_func)
        cost = cost_function(acts["A{}".format(len(network_dims)-1)], train_y)
        d = backward_propagation(params, zs, acts, train_x, train_y, network_dims, layer_act_func, final_act_func)
        update_params(d, params, learning_rate, network_dims)
        print('cost after iteration: {}: {}'.format(i, cost))
    return

if __name__ == "__main__":
    exit(main())



