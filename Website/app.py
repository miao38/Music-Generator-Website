from flask import Flask, render_template, redirect, url_for, send_file
import Music_Generator_2
from Tokenizer import *

app = Flask(__name__)

@app.route('/static/')
@app.route('/static/generate')
@app.route('/')
def start():
    return render_template('index.html')

@app.route('/generated')
def generated():
    music_file = Music_Generator_2.starter("no") #from Music_Generator_2
    return send_file(music_file, mimetype="audio/midi")

app.run(debug=True)
