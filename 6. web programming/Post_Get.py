# -*- coding: utf-8 -*-
'''
https://hackmd.io/@shaoeChen/HJkOuSagf?type=view
Goal: using Get and Post command with python in flask

in cmd 
    >python Post_Get.py runserver

in browser
    http://127.0.0.1:5000/login
'''

from flask import Flask, request

# In[1]

app = Flask(__name__)

@app.route('/login', methods=['GET', 'POST']) 
def login():
    if request.method == 'POST': 
        return 'Hello ' + request.values['username'] 

    return "<form method='post' action='/login'><input type='text' name='username' />" \
            "</br>" \
           "<button type='submit'>Submit</button></form>"

# In[2]

if __name__ == '__main__':
    app.debug = True
    app.run()    