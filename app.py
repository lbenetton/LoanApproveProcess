from flask import Flask, redirect, url_for, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# load saved artifacts
def load_model():
    return pickle.load(open('artifacts/loan_approve.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/notebook')
def notebook():
    return render_template('notebook.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        gender = request.form['inlineRadioGender']
        married = request.form['inlineRadioMarried']
        dependents = request.form['inlineRadioDependents']
        education = request.form['inlineRadioEducation']
        selfemp = request.form['inlineRadioSelfEmp']        
        credit = request.form['inlineRadioCredit']
        proparea = request.form['inlineRadioPropArea']
        appincome = float(request.form['textAppIncome'])
        coappincome = float(request.form['textCoAppIncome'])
        loanamt = float(request.form['textLoanAmt'])
        loanterms = int(request.form['textLoanTerms'])

        x = np.zeros(20)

        x[0] = appincome
        x[1] = coappincome
        x[2] = loanamt
        x[3] = loanterms
        
        credithistory = credit
        if credithistory == 'yes':
            x[4] = 1
        elif credithistory == 'no':
            x[4] = 0

        gender = gender
        if gender == 'male':
            x[5],x[6] = 0, 1
        elif gender == 'female':
            x[5],x[6] = 1, 0
        else:
            x[5],x[6] = 1, 0

        married = married
        if married == 'notmarried':
            x[7],x[8] = 1, 0
        elif married == 'married':
            x[7],x[8] = 0, 1
        else:
            x[7],x[8] = 0, 1

        dependents = dependents
        if dependents == '0':
            x[9],x[10],x[11],x[12] = 1, 0, 0, 0
        elif dependents == '1':
            x[9],x[10],x[11],x[12] = 0, 1, 0, 0
        elif dependents == '2':
            x[9],x[10],x[11],x[12] = 0, 0, 1, 0
        elif dependents == '3+':
            x[9],x[10],x[11],x[12] = 0, 0, 0, 1

        education = education
        if education == 'graduate':
            x[13],x[14] = 1, 0
        elif education == 'notgraduate':
            x[13],x[14] = 0, 1

        selfemployed = selfemp
        if selfemployed == 'no':
            x[15],x[16] = 0, 1
        elif selfemployed == 'yes':
            x[15],x[16] = 1, 0

        propertyarea = proparea
        if propertyarea == 'rural':
            x[17],x[18],x[19] = 1, 0, 0
        elif propertyarea == 'semiurban':
            x[17],x[18],x[19] = 0, 1, 0
        elif propertyarea == 'urban':
            x[17],x[18],x[19] = 0, 0, 1

        x = x.reshape(1,20)

        model = load_model()
        prediction = model.predict(x)

        labels = ['Loan Rejected', 'Loan Accepted']
        result = labels[prediction[0]]

        return render_template('predict.html', output='{}'.format(result))
    else:
        return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)