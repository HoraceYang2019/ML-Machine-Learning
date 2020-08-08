# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:33:39 2017

@author: hao
"""

import socket
import sys
import numpy as np

# Create a TCP/IP socket
PortNo = 10000
#IPaddr = '192.168.1.36'
IPaddr = '127.0.0.1'
MaxBuf = 128
message = ' I am the King!'

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# In[]
# Bind the socket to the port
server_address = (IPaddr, PortNo)
print(sys.stderr, 'starting up on %s port %s' % server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

def fun(x):
    a, b, c = 1, 1, 1
    y = a*x*x+b*x+c
    return str(y)

# In[]
while True:
    # Wait for a connection
    print(sys.stderr, 'waiting for a connection')
    connection, client_address = sock.accept()
    
    try:
        print(sys.stderr, 'connection from', client_address)

        # Receive the data in small chunks and retransmit it
        while True:
            rdata = connection.recv(MaxBuf)
            #print(sys.stderr, 'received: '+ bytes.decode(rdata))
            #data = sqrt(np.float32(bytes.decode(rdata)))
            
            if rdata:
                result=fun(float(rdata))
                print(sys.stderr, 'sending data back to the client')
               # data = data + message.encode(encoding='utf-8')
                connection.sendall(result.encode(encoding='utf-8'))
            else:
                print(sys.stderr, 'no more data from', client_address)
                break
            
    finally:
        # Clean up the connection
        connection.close()