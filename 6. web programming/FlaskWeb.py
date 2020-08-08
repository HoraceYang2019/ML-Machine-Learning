'''
https://blog.techbridge.cc/2017/06/03/python-web-flask101-tutorial-introduction-and-environment-setup/
Goal: a new flask web 

in cmd 
    >python FlaskWeb.py runserver

in browser
    http://127.0.0.1:5000/
    or 
    http://127.0.0.1:5000/cakes
'''

from flask import Flask

# In[1]
# Initial Flask class as an instance
app = Flask(__name__)

# mapping route with index function
@app.route('/')
def index():
    return 'Hello Horace'

# routing to other path
@app.route('/cakes')
def cakes():
    return 'Yummy cakes!'

# In[2]
# 判斷自己執行非被當做引入的模組，因為 __name__ 這變數若被當做模組引入使用就不會是 __main__
if __name__ == '__main__':
#    app.run(debug=True, host='127.0.0.1')
    app.run(host='127.0.0.1')
