from flask import Flask, request, render_template
import sys
import pygit2
import re

app = Flask(__name__)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route('/')
def index():
    repo=pygit2.Repository(r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code")
    diff = repo.diff().patch
    print(diff)
    return render_template('index.html', commitDiff = diff)
