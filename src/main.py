import numpy as np
import pandas as pd
from base_line import *
from gradient_Descent import *
from metrics import *
from normalize_equation import *

data = pd.read_csv('student.csv')

math = data['Math'].values
reading= data['Reading'].values
writing = data['Writing'].values

qut_samples = len(math) # quantidade de amostras na base de treino 
x0 = np.ones(qut_samples) # x0 acompanha theta0, sempre igual a 1
X = np.array([x0,math, reading])
theta = np.zeros((1,3))
Y = np.array(writing)
Y = Y.reshape(1,qut_samples)
alpha = 0.0001
iterations = 1000

newTheta,cost_history = gradient_descent(X,Y,theta, alpha, iterations)
newTheta_norm = normalize_equation(X,Y,theta)
Y_pred_norm = newTheta_norm.T.dot(X)
Y_pred = newTheta.dot(X)
rmse_norm = rmse(Y,Y_pred_norm)
rmse_ = rmse(Y,Y_pred)
print("RSME error: ",rmse_)
print("RSME error norm: ", rmse_norm)


