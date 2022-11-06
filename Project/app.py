from flask import Flask, escape, request, render_template
import pickle, requests, json

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

        # api request call
        url = 'https://newsapi.org/v2/everything?q={}&apiKey=8535a18a149149e59842155bfa13ec8d'.format(news)
        r = requests.get(url)
        data = json.loads(r.content)
        urlLink = []
        for i in range(len(data['articles'])):
            urlLink.append(data['articles'][i]['url'])
        # api call end

        if predict == 'REAL':
            class_name = "real"        
        else:
            class_name = "fake"

        return render_template("prediction.html",prediction_text = "The News headline is predicted as {}".format(predict),class_name = class_name,url_list = urlLink)

    else:
        return render_template("prediction.html")

if __name__ == '__main__':
    app.run()