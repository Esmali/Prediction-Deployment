from flask import Flask, request, Response
import jsonpickle
import numpy as np
from test import prediction

app = Flask(__name__)


# default flask api home page
@app.route('/')
def index():
    return "Hello, World! Version 4"


# route http posts to this method
@app.route('/predict', methods=['POST'])
def test():
    data = request.get_json()
    pred = prediction(data)
    response_pickled = jsonpickle.encode(pred)
    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
if __name__ == "__main__":
    app.run(host='localhost', port=8080)
