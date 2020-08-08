# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:47:29 2019

@author: ASUS
"""

import os
import logging

class CommunicationLogFileHandler(logging.Handler):
    def __init__(self, path, prefix=""):
        logging.Handler.__init__(self)

        self.path = path
        self.prefix = prefix

    def emit(self, record):
        filename = os.path.join(self.path, "{}com_{}.log".format(self.prefix, record.remoteName))
        with open(filename, 'a') as f:
            f.write(self.format(record) + "\n")