from flask import Flask,request,render_template
import pandas
import numpy as np
import pickle



model=pickle.load(open("model.pkl","rb"))




# flask app
app= Flask(__name__)





@app.route('/', methods=['GET', 'POST'])  ## it is used to give path of index.html file
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    featurs = request.form['feature']
    
    featurs_lst = featurs.split(',')
    np_features = np.asarray(featurs_lst,dtype=np.float32).reshape(1, -1)
    pred= model.predict(np_features)


    output= ["Cancrouse" if pred[0]== 1 else "Not Cancrouse"]


    return render_template('index.html',message=output)





#python main
if __name__ == "__main__":
    app.run(debug = True)