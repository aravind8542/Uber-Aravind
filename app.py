import numpy as np
from flask import Flask ,request,render_template
import pickle
import math

app=Flask(__name__)

model=pickle.load(open('taxi.pkl','rb'))


#add the  decorator to link the html page

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=round(prediction[0],2)


    return render_template('index.html',prediction_text="number of weekly Rides should be {}".format(math.floor(output)))

if __name__=='__main__':
    app.run(debug =True)
