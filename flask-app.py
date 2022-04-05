from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import joblib
import yaml
# from prediction_service import prediction


params_path = "params.yaml"
web_root = "flask-app"

static_dir = os.path.join(web_root, "static")
template_dir = os.path.join(web_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder = template_dir)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config



def predict(data):
    config = read_params(params_path)
    model_dir_path = config["app model dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)
    print(prediction)
    return prediction

def api_response(request):
    pass


@app.route("/", methods=["GET","POST"])    # renders template
def index():
    if request.method == "POST":
        try:
            if request.form:
                data = (request.form).values()
                data = [list(map(float, data))]     # creates 2-D list
                response = predict(data)
                return render_template("index.html", response = response)

            elif request.json:                      # query model from another app or with POST method   
                response = api_response(request)
                return jsonify(response)


        except Exception as e:
            print(e)
            error = {"error": "Something went wrong. Try Again!!"}
            return render_template("404.html", error = error)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
