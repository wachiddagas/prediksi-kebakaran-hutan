from flask import Flask, render_template
from markupsafe import escape

app = Flask(__name__)

@app.route('/')
def index():
    return "ini adalah index"

@app.route('/<name>')
def hello(name):
    x = 12
    x = 12*name
    return f"Hello world, {x}"

@app.route('/hola')
def hola():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/hitungan')
def hitungan():
    title = "Prediksi Kebakaran Hutan"
    return render_template('hitungan.html', title=title)