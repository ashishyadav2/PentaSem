from pydoc import classname
from flask import Flask, escape, request, render_template
import pickle

vector = pickle.load(open("vectorizer.pkl",'rb'))

model = pickle.load(open("finalized_model.pkl",'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        predict = model.predict(vector.transform([news]))[0]
 
        if predict == 'REAL':
            class_name = "real"        
        else:
            class_name = "fake"

        return render_template("prediction.html",prediction_text = "The News headline is predicted as {}".format(predict),class_name = class_name)

    else:
        return render_template("prediction.html")

if __name__ == '__main__':
    app.run()