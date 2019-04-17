# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:39:43 2019

@author: dell
"""

from PIL import Image
import numpy as np

a = np.asarray(Image.open('./test001.png').convert('L')).astype('float')

im = Image.fromarray(a.astype('uint8')) 	#重构图像
im.save('./test001.png')