#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:15:02 2021

@author: azubaidi Hi arkan

"""

#path='/data2/azubaidi/github/ANCP/voxelwiseencoding/tests/'

#path='/data2/azubaidi/github/ANCP/voxelwiseencoding/voxelwiseencoding/'

#import tests.test_encoding 

from tests.test_encoding import test_encoding_synthethic_data

models, score_list, target_prediction, y, X, train_indices, test_indices = \
    test_encoding_synthethic_data('/data2/azubaidi/github/ANCP/voxelwiseencoding'
                                  +'/DownTheRabbitHoleFinal_mono_exp120_NR16_pad.tsv.gz')