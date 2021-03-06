# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import sys
import keras
import keras.backend as K
from keras.models import Sequential, Model

class CollaborativeModel(object):
    def __init__(self, config):
        self._name = "CollaborativeModel"
        self.config = {}
        self.check_list = []

    def set_default(self, k, v):
        if k not in self.config:
            self.config[k] = v

    def setup(self, config):
        pass

    # list of parameters required by a model
    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print(e, end='\n')
                print('[Model] Error %s not in config' % e, end='\n')
                return False
        return True

    def build(self):
        pass

    def check_list(self, check_list):
        self.check_list = check_list
