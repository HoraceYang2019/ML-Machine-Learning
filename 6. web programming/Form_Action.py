# -*- coding: utf-8 -*-
'''
https://hackmd.io/@shaoeChen/HJkOuSagf?type=view
Goal: reroute the web pages

new login.html and hello.html in templates folder

in cmd 
    >python Form_Action.py runserver

in browser
    http://127.0.0.1:5000/login
'''

from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# In[10]

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('hello', username2=request.form.get('username1')))

    return render_template('login.html')

# re-route to this page 
@app.route('/hello/<username2>')
def hello(username2):
    return render_template('hello.html', username3=username2)

# In[50]
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)