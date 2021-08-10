"""
Date: 2021-08-09 12:37:15
LastEditors: GodK
"""

import numpy as np



def NN_F2(traindata):
    """[summary]

    Args:
        traindata (ndarray): shape:[input_dimension, trainlength]
    """
    layer_nodes_num = [400, 400, 300, 150]
    input_dimension, trainlength = traindata.shape
    pre_layer_nodes_num = input_dimension

    all_weights = {}
    for l in range(len(layer_nodes_num)):
        curr_layer_nodes_num = layer_nodes_num[l]
        for it in range(curr_layer_nodes_num):
            curr_node_input_weight = np.random.randn(pre_layer_nodes_num)
            all_weights[l, it] = curr_node_input_weight
        pre_layer_nodes_num = curr_layer_nodes_num

    output = np.zeros((layer_nodes_num[-1], trainlength))
    for i in range(trainlength):
        pre_layer_nodes_num = input_dimension
        pre_layer_nodes_value = traindata[:, i]
        for l in range(len(layer_nodes_num)):
            curr_layer_nodes_num = layer_nodes_num[l]
            curr_layer_nodes_value = np.zeros(curr_layer_nodes_num)
            for it in range(curr_layer_nodes_num):
                curr_node_input_weight = all_weights[l, it]

                xx = np.sum(pre_layer_nodes_value * curr_node_input_weight.T)
                curr_layer_nodes_value[it] = np.tanh(xx / 2.5)
            pre_layer_nodes_value = curr_layer_nodes_value.T
            pre_layer_nodes_num = curr_layer_nodes_num
        output[:, i] = curr_layer_nodes_value
    return output