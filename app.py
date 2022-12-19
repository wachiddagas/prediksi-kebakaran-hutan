from flask import Flask, render_template, Response, url_for
import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

app = Flask(__name__)
title = "Prediksi luas area terdampak"
subject = "Berikut adalah sistem prediksi area terdampak kebakaran hutan"

@app.route("/")
def hello_world():
   
    return render_template("homepage.html", title=title, subject=subject)

@app.route("/hitung")
def hitung():

    data_aceh = pd.read_excel("static/aceh.xlsx", "aceh", header=0, parse_dates=True)
    print(data_aceh)
    ds = data_aceh.index.to_series()

    mask_aceh = np.random.rand(len(data_aceh)) < 0.5
    df_train_aceh = data_aceh[mask_aceh]
    df_test_aceh = data_aceh[~mask_aceh]
    print('Training data set length='+str(len(df_train_aceh)))
    print('Testing data set length='+str(len(df_test_aceh)))
    expr = """ luas ~ landsat8 + noaa20 + snpp + terraaqua """
    y_train, X_train = dmatrices(expr, df_train_aceh, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test_aceh, return_type='dataframe')
    poisson_training_hasil_aceh = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_hasil_aceh.summary())
    #Make some predictions on the test data set.
    poisson_predictions_aceh = poisson_training_hasil_aceh.get_prediction(X_test)
    #.summary_frame() returns a pandas DataFrame
    predictions_summary_frame_aceh = poisson_predictions_aceh.summary_frame()
    print(predictions_summary_frame_aceh)

    predicted_counts_aceh=predictions_summary_frame_aceh['mean']
    actual_counts = y_test['luas']

    #Mlot the predicted counts versus the actual counts for the test data.
    fig = plt.figure()
    fig.suptitle('Prediksi dan Kenyataan Luas terdampak kebakaran hutan')
    predicted, = plt.plot(X_test.index, predicted_counts_aceh, 'go-', label='Predicted counts')
    actual, = plt.plot(X_test.index, actual_counts, 'ro-', label='Actual counts')
    plt.legend(handles=[predicted, actual])
    plt.show()

    #Show scatter plot of Actual versus Predicted counts
    plt.clf()
    fig = plt.figure()
    fig.suptitle('Scatter plot of Actual versus Predicted counts')
    plt.scatter(x=predicted_counts_aceh, y=actual_counts, marker='.')
    plt.xlabel('Predicted counts')
    plt.ylabel('Actual counts')
    plt.show()
    fig3 = plt.show()
    
    return render_template("hitung_aceh.html",title=title, subject=subject, fig3=fig3)

@app.route("/hitung2")
def hitung2():

    data_aceh = pd.read_excel("static/REGRESI LINIER BERGANDA 4.xlsx", "Sheet2", header=0, parse_dates=True)
    print(data_aceh)
    ds = data_aceh.index.to_series()

    mask_aceh = np.random.rand(len(data_aceh)) < 0.9
    df_train_aceh = data_aceh[mask_aceh]
    df_test_aceh = data_aceh[~mask_aceh]
    print('Training data set length='+str(len(df_train_aceh)))
    print('Testing data set length='+str(len(df_test_aceh)))
    expr = """ Y ~ X1+X2+X3+X4+X5+X6+X7+X8+X9 """
    y_train, X_train = dmatrices(expr, df_train_aceh, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test_aceh, return_type='dataframe')
    poisson_training_hasil_aceh = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_hasil_aceh.summary())
    #Make some predictions on the test data set.
    poisson_predictions_aceh = poisson_training_hasil_aceh.get_prediction(X_test)
    #.summary_frame() returns a pandas DataFrame
    predictions_summary_frame_aceh = poisson_predictions_aceh.summary_frame()
    print(predictions_summary_frame_aceh)

    predicted_counts_aceh=predictions_summary_frame_aceh['mean']
    actual_counts = y_test['Y']

    #Mlot the predicted counts versus the actual counts for the test data.
    fig = plt.figure()
    fig.suptitle('Prediksi dan Kenyataan Luas terdampak kebakaran hutan')
    predicted, = plt.plot(X_test.index, predicted_counts_aceh, 'go-', label='Predicted counts')
    actual, = plt.plot(X_test.index, actual_counts, 'ro-', label='Actual counts')
    plt.legend(handles=[predicted, actual])
    plt.show()

    #Show scatter plot of Actual versus Predicted counts
    plt.clf()
    fig = plt.figure()
    fig.suptitle('Scatter plot of Actual versus Predicted counts')
    plt.scatter(x=predicted_counts_aceh, y=actual_counts, marker='.')
    plt.xlabel('Predicted counts')
    plt.ylabel('Actual counts')
    plt.show()
    fig3 = plt.show()
    
    return render_template("homepage.html",title=title, subject=subject, fig3=fig3)

if __name__ == "__main__":
    app.run(debug=True)