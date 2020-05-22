from flask import Flask, jsonify, request, render_template
from wms import predict, delete_path, gen_chunks
import random

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def render():
    if request.method == 'POST':
       text = request.form['yt_url']
       url = str(text)
       substring = 'youtube.com'
       if substring in url:
           head, sep, tail = url.partition('youtube.com/watch?v=')
           url = tail
           head, sep, tail = url.partition('&')
           url = head
       else:
           head, sep, tail = url.partition('.be/')
           url = tail 
       laughPercent, numLaughs, laughsPerMin, laughTimesList, laughTimes = predict(url)
       f = open("bucket/analysis.txt", "r")
       analysis = f.readlines()
       analysis_line = random.choice(analysis)
       return render_template('form.html', _input = True,  _laughPercent = laughPercent, 
               _numLaughs = numLaughs, _laughsPerMin = laughsPerMin,
               _laughTimesList = laughTimesList, _url = url, _laughTimes = laughTimes, _analysis = analysis_line)

    return render_template('form.html')
#@app.route("/", methods =['POST'])
#def pred():
#    text = request.form['yt_url']
#    url = str(text)
#
#    predictions = predict(url)
#    return predictions

if __name__== "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
