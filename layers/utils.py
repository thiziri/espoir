# -*- coding=utf-8 -*-
import os
import sys
import traceback
import psutil


def show_layer_info(layer_name, layer_out):
    print('[layer]: %s\t[shape]: %s \n%s' % (layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))


def show_memory_use():
    used_memory_percent = psutil.virtual_memory().percent
    strinfo = '{}% memory has been used'.format(used_memory_percent)
    return strinfo