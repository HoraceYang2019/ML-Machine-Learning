# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:30:43 2019
Goal: create RESTful API
https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask

in cmd 
    >python WebService.py runserver

in browser
    http:\\127.0.0.1:5000/todo/api/v1.0/tasks

@author: ASUS
"""

#!flask/bin/python
from flask import Flask, jsonify
from flask import abort

# In[1]
# Initial Flask class as an instance
app = Flask(__name__)

tasks = [
        {
                'id': 1,
                'title': u'Buy groceries',
                'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
                'done': False
                },
         {
                 'id': 2,
                 'title': u'Learn Python',
                 'description': u'Need to find a good Python tutorial on the web',
                 'done': True
                 }
         ]

# In[20]: using 'Get' command to access data from web server
# http:ip:5000/todo/api/v1.0/tasks

@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'TASKS': tasks})

# In[21]: http:ip:5000/todo/api/v1.0/tasks/2
    
@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = list(filter(lambda t: t['id'] == task_id, tasks))
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})

# In[30]: using 'Post' command to send data to web server

from flask import request
@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)
    task = {
        'id': tasks[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': task}), 201

# In[50]

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)