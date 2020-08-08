# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:45:03 2019

@author: ASUS
"""

import logging
import code
import secsgem
from communication_log_file_handler import CommunicationLogFileHandler

# In[]

class SampleHost(secsgem.GemHostHandler):

    def __init__(self, address, port, active, session_id, name, custom_connection_handler=None):

        secsgem.GemHostHandler.__init__(self, address, port, active, session_id, name, custom_connection_handler)

        self.MDLN = "gemhost"
        self.SOFTREV = "1.0.0"

# In[]

commLogFileHandler = CommunicationLogFileHandler("log", "h")
commLogFileHandler.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
logging.getLogger("hsms_communication").addHandler(commLogFileHandler)
logging.getLogger("hsms_communication").propagate = False
logging.basicConfig(format='%(asctime)s %(name)s.%(funcName)s: %(message)s', level=logging.DEBUG)

h = SampleHost("127.0.0.1", 5000, False, 0, "samplehost")
h.enable()

code.interact("host object is available as variable 'h'", local=locals())

h.disable()
