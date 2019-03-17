#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:09:58 2019

@author: fujimoto
"""

from PIL import Image
import glob

files = sorted(glob.glob('./sample_images_while_training/celeba_ch_mask_output10/*.png'))
images = list(map(lambda file: Image.open(file), files))

images[0].save('./sample_images_while_training/celeba_ch_mask_output10/example.gif', save_all=True, append_images=images[1:], duration=66, loop=0)