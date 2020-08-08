# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 07:52:24 2017

@author: ASUS
"""
Installatoin:
pip install paho-mqtt

Example:
    1. TCPClient & TCPServer
    2. Subscriber & Publisher
    3. Backend_dT.py:send timer value by MQTT
    4. GEM_host & GEM_Eqp
    
----------------------------------------------------------------------------------------------
Installatoin of Node-red:

1. install Node.js, node-red
    -----------------------------------------------------------------------
    for Rasbian: 
        sudo apt-get update
        sudo apt-get upgrade
        sudo apt-get install nodejs npm
        sudo npm install -g --unsafe-perm node-red
        
        install nodes 
        cd ~/.node-red
        sudo node-red-stop
        sudo npm install node-red-dashboard
        sudo node-red-start
    
    Ref:
    
    node-red -s D:\Dropbox\Codes\Python\RaspberryPi\Node-Red\settings.js -u D:\Dropbox\Codes\Python\RaspberryPi\Node-Red
    -----------------------------------------------------------------------
    for Windows:
        npm install -g --unsafe-perm node-red

    
2. run node-red in command mode
    path: command mode > node-red
          browser > http://127.0.0.1:1880

3. run Calculator REST Service
https://www.youtube.com/watch?v=xntpcrLoeyE&feature=youtu.be
    

4. install nodes in node-red
    path: from palette-> user-setting->install->palette search for 
    item: node-red-dashboard, input-split 

5. import flows
    path:  import-> library 
    item: Control.json, Monitor.json

---------------------------------------------------------------------------------------------------    