'''
-----------------------------------------------------------------------
for Rasbian: 
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install nodejs npm
    sudo npm install -g --unsafe-perm node-red

Ref:
-----------------------------------------------------------------------
for Windows:
    npm install -g --unsafe-perm node-red
-----------------------------------------------------------------------
@author: Horace
'''
import paho.mqtt.publish as publish
import json

import time
import threading
import datetime

host = "broker.hivemq.com"
portNo = 	1883

msg = {"subject":"Math","score":70}
jmsg = json.dumps(msg, ensure_ascii=False)

# In[]

smpPeriod = 3     #sample period in sec
#--------------------------------------------------------------------------------------------
# Enable a stable time to triger data collection
#--------------------------------------------------------------------------------------------            
class T0(object):
    
    def __init__(self, period):
        self.pd = period
    
    def run(self):
        ts = datetime.datetime.now()
        ts = time.mktime(ts.timetuple())
        while (True):
            tf = datetime.datetime.now()
            tf = time.mktime(tf.timetuple())
            msg = str(tf-ts)
            jmsg = json.dumps(msg, ensure_ascii=False)
            publish.single("mislab/id/xyz", jmsg, hostname = host)
            ts=tf
            time.sleep(self.pd)

#------------------------------------------------------------------------------------------------        
if __name__ == '__main__':

    t0 = T0(smpPeriod)
    mt = threading.Thread(target = t0.run,  args=())

    mt.start()  # start the triger timer
