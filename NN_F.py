"""
Date: 2021-08-09 12:36:17
LastEditors: GodK
"""
import numpy as np


def NN_F(input):
    """[summary]

    Args:
        input (ndarray):
    """
    layer_nodes_num = [200, 200, 150, 150]
    bias = 0.00001
    pre_layer_nodes_num = len(input)
    pre_layer_nodes_value = input.T
    for l in range(len(layer_nodes_num)):
        curr_layer_nodes_num = layer_nodes_num[l]
        curr_layer_nodes_value = np.zeros(curr_layer_nodes_num)
        for it in range(curr_layer_nodes_num):
            curr_node_input_weight = np.random.randn(pre_layer_nodes_num)
            xx = np.sum(pre_layer_nodes_value*curr_node_input_weight)
            curr_layer_nodes_value[it] = np.tanh(xx/2.5)
        pre_layer_nodes_value=curr_layer_nodes_value
        pre_layer_nodes_num=curr_layer_nodes_num
    
    output=curr_layer_nodes_value
    return output