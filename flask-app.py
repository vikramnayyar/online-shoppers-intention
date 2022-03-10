from flask import Flask, render_template, request, jsonify
import os
import numpy as np
# from prediction_service import prediction


params_path = "params.yaml"
web_root = "flask-app"

static_dir = os.path.join(web_root, "static")
template_dir = os.path.join(web_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder = template_dir)

@app.route("/", methods=["GET","POST"])    # renders template
def index():
    if request.method == "POST":
        pass
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
