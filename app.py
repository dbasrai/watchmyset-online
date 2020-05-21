from flask import Flask, jsonify, request, render_template
from wms import predict, delete_path, gen_chunks
import random 

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def render():
    if request.method == 'POST':
       text = request.form['yt_url']
       url = str(text)
       laughPercent, numLaughs, laughsPerMin, laughTimesList = predict(url)
       f = open("bucket/analysis.txt", "r")
       analysis = f.readlines()
       analysis_line = random.choice(analysis)
       return render_template('form.html', _input = True,  _laughPercent = laughPercent, 
               _numLaughs = numLaughs, _laughsPerMin = laughsPerMin,
               _laughTimesList = laughTimesList,_analysis = analysis_line)

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
