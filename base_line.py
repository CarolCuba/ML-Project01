import numpy as np
from sklearn import linear_model
from sklearn import metrics

'''
Funcao para calcular regressao linear utilizando sklearn 
	Input:
	    X: matriz de features
	    Y: matriz de gabarito
	    alpha: learning rate
	    iterations: numero de iteracoes 
	Output:
	    rmse: valor de erro gerado na predicao
		

'''
def baseLine_GD(X,Y,alpha,iterations):
    Y = Y.T
    X = X.T
    reg = linear_model.LinearRegression()
    reg.fit(X,Y)
    Y_predpred = reg.predict(X)
    rmse = np.sqrt(metrics.mean_squared_error(Y.T, Y_pred))
    return rmse

    
