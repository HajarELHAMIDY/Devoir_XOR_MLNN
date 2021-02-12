# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 02:12:17 2021

@author: EL-HAMIDY
"""

import numpy as np



def initialize_parameters_random(input_dims,hidden_dims,output_dims):
    hidden_weights=np.random.uniform(size=(input_dims, hidden_dims))
    hidden_bias=np.random.uniform(size=(1,hidden_dims))
    output_weights=np.random.uniform(size=(hidden_dims, output_dims))
    output_bias=np.random.uniform(size=(1, output_dims))

    return [hidden_weights,hidden_bias,output_weights,output_bias]

