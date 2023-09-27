from flask import Flask,render_template,request
import pickle
applicaton=Flask(__name__)
app=applicaton

@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='GET':
        return render_template('index.html')
    else:
        type_=request.form.get('Type')
        Air_temp=request.form.get('Air_temp')
        Process_temp=request.form.get('Process_temp')
        Rotational_speed=request.form.get('Rotational_speed')
        Torque=request.form.get('Torque')
        Tool_wear=request.form.get('Tool_wear')
        TWF=request.form.get('TWF')
        HDF=request.form.get('HDF')
        PWF=request.form.get('PWF')
        OSF=request.form.get('OSF')
        if type_=='M':
            type_=2
        elif type_=='L':
            type_=1
        else:
            type_=0

        def predict_machine_failure(model,Type,Air_temp,Process_temp,Rotational_speed,Torque,Tool_wear,TWF,HDF,PWF,OSF):
            if (TWF==1) or (HDF==1) or (PWF==1) or (OSF==1):
                return True
            else:
                predicted=model.predict([[Type,Air_temp,Process_temp,Rotational_speed,Torque,Tool_wear]])
            return True if predicted[0]==1 else False

        loaded_model = pickle.load(open('random_forest_model.pkl', 'rb'))
        predicted_result=predict_machine_failure(
            loaded_model,type_,Air_temp,Process_temp,Rotational_speed,Torque,
            Tool_wear,TWF,HDF,PWF,OSF
        )
        if predicted_result is True:
            output='Machine Failed'
        else:
            output='Machine not Failed'

        return render_template('index.html',result=output)
if __name__=='__main__':
    app.run(host='0.0.0.0')

