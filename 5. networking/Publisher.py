# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 21:56:59 2017
https://mosquitto.org/download/
@author: ASUS
"""

# Publisher.py
import paho.mqtt.publish as publish
import json

#host = '127.0.0.1'
host = "broker.hivemq.com"
portNo = 	1883

msg = {"subject":"Math","score":70}
jmsg = json.dumps(msg, ensure_ascii=False)

# In[]
# Test 1: start later to publish mqtt message by "mislab/mqtt/simple" 

publish.single("mislab/mqtt/simple", jmsg, hostname= host)

# In[]
# Test 2: start later to publish mqtt message by "mislab/mqtt/callback" 

publish.single("mislab/mqtt/control", jmsg, hostname = host)

# In[]
# Test 3: start later to publish mqtt message by "mislab/mqtt/multiple" 

msgs = [{'topic':"mislab/mqtt/multiple",'payload': "multiple 1"},
        ("mislab/mqtt/multiple", 'multiple 2', 0, False)]
publish.multiple(msgs, hostname = host)

# In[]
# Test 4: use another MQTT Server
# start later to publish mqtt message by 'mislab/mqtt/callback' 

host = "m14.cloudmqtt.com" 
portNo = 	17640 
authpass = {'username':"vfhmwuwd", 'password':"9Na3SdDn7KvW"}

publish.single("mislab/mqtt/cloudmqtt", jmsg, hostname= host, port = portNo, auth = authpass)