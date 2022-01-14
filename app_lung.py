import os
from flask import Flask,jsonify,request,render_template,redirect
from lung import predict
from werkzeug.utils import secure_filename

app = Flask(__name__) ## __name__= current file name  (main)

@app.route("/", methods = ["GET", "POST"]) ## page name 
def index():
    prediction = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            app.config['Audio_UPLOADS'] = ""
            file = request.files['file']
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["Audio_UPLOADS"], filename))
            actual_file = filename

            prediction = predict(actual_file)
        
        print(prediction)

    return render_template('index.html', prediction=prediction)


app.run(debug=True, threaded = True)