# -*- coding: utf-8 -*-
'''
https://hackmd.io/@shaoeChen/HJkOuSagf?type=view

create folder named templates
new abc.html

in cmd 
    >python Render_Template.py runserver

in browser
    http://127.0.0.1:5000/para/[cmd]

'''

from flask import Flask
from flask import render_template

# In[1]
app = Flask(__name__)

@app.route('/para/<user>')
def index(user):
    return render_template('abc.html', user_template=user)

# In[2]

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)