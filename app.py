from flask import Flask, render_template,request
import joblib
import numpy as np
#global object
model=joblib.load('heart_disease prediction_saved')
sc=joblib.load('scaling_ob')

app = Flask(__name__)

@app.route('/')
def credentials():
    return render_template('login.html',msg='Enter details')

@app.route('/criteria',methods=["POST"])
def eligibilty():
    msg='invalid password ..try again'
    pswd=request.form['pswd']
    usr=request.form['usr']
    #print(pswd)
    if(pswd=='sklearn'):
        return render_template('new.html',usr=usr)
    
    else:     
        return render_template('login.html',msg=msg)

@app.route('/prediction',methods=["POST"])
def getData():
    
    feature=[]
    for x in request.form.values():
        feature.append(float(x))
   
    arr=[np.array(feature,dtype=np.float32)]    

    
    arr=sc.transform(arr)
    res=int(model.predict(arr) )
    
    if(res==1):
        label="serious"
    else:
        label="not serious"
    res1=model.predict_proba(arr)
    prob=max(res1[0][0],res1[0][1])
    
    return render_template('recieved.html',label=label,label2="{0:.2f}".format(prob*100) )


@app.route('/description')
def description():
    return 'Machine learning algorithm for CHD disease seriousness detection'

if __name__ == "__main__":
    app.run(port=8000)

    


  