from flask import Flask, jsonify, request
from wms import predict, delete_path, gen_chunks

app = Flask(__name__)
@app.route("/")
def hello():
    predictions = predict('https://www.youtube.com/watch?v=45udmQIicTk')
    return predictions

if __name__== "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
