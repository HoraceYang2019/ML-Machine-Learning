# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 21:51:19 2017

@author: ASUS
"""
# Subscriber.py
import paho.mqtt.subscribe as subscribe

#host = '127.0.0.1'
host = "broker.hivemq.com"
portNo = 	1883

def on_message_print(client, userdata, message):
    print("%s %s" % (message.topic, message.payload))

# In[]
# Test 1: start first to subscribe mqtt message by "mislab/mqtt/simple" 
    
msg = subscribe.simple("mislab/mqtt/simple", hostname=host)
print("I got mqtt: %s with %s" % (msg.topic, msg.payload))

# In[]
# Test 2: start first to subscribe mqtt message by "mislab/mqtt/callback" without stop

subscribe.callback(on_message_print, "mislab/id/xxx", hostname=host)

# In[]
# Test 3: start first to subscribe mqtt message by "mislab/mqtt/multiple" without stop

subscribe.callback(on_message_print, "mislab/mqtt/multiple", hostname=host)

# In[]
# Test 4: use another MQTT Server

host = "m14.cloudmqtt.com" 
portNo = 	17640 
authpass = {'username':"vfhmwuwd", 'password':"9Na3SdDn7KvW"}
subscribe.callback(on_message_print, "mislab/mqtt/cloudmqtt", hostname=host, port = portNo, auth = authpass)