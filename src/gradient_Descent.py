import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn import metrics



'''Criando a funcao para normalização dos valores de um Dataframe
    input:
        df: Dataframe
    output:
        df: Dataframe normalizado
'''
def normalize_dataframe(df):
    for column in df:
        df[column] = (df[column] - min(df[column])) / ( max(df[column]) - min(df[column]) )
    return df


'''Criando a funcao para normalização de colunas especificas de um Dataframe
    input:
        df: Coluna de um Dataframe
    output:
        df: Coluna de Dataframe normalizada
'''
def normalize_dataframe_column(df):
        return (df - min(df)) / ( max(df) - min(df) )

def stochastic_gradient_descent(X,Y,theta,alpha,iterations):
    qut_samples = Y.shape[0]
    cost_history = [0]*iterations*qut_samples
    
    for iteration in range (iterations):
        for sample in range (qut_samples):#hipotese
            h = X.iloc[sample,:].dot(theta)
            #diferenca entre hipotese e predicao
            loss = h - Y[sample]
            gradient = X.iloc[sample,:].T * loss
            #Gradiente descendente
            #gradient = np.array(loss * X.iloc[:,sample].T).reshape(1, theta.shape[1])
            #changing values of parameters 
            theta = theta - np.array([(alpha * gradient)]).T
            #New cost value
            cost = cost_function(X,Y,theta)
            #print ("Iteration: %d | Cost: %f" % (iteration+1, cost))
            cost_history[iteration*qut_samples+sample] = cost
    return theta, cost_history


'''Criando a funcao de custo
    input:
        X: matriz de features
        Y: matriz de gabarito
        theta: array de pesos
    output:
        custo: custo da iteração
'''
def cost_function(X,Y,theta):
    qut_samples = Y.shape[0]
    h = X.dot(theta)
    cost = np.sum((h - Y)**2)/(2*qut_samples)
    return cost

'''Criando funcao para executar o gradiente descendente (batch)
    input:
        X: matriz de features
        Y: matriz de gabarito
        theta: array de pesos
        alpha:learning rate
        iterations: numero max de iteracoes para parada
    output:
       theta: vetor de thetas atualizados
       cost_history: vetor do historico de custos ao longo do processo
'''
def gradient_descent(X,Y,theta,alpha,iterations):
    cost_history = [0]*iterations
    qut_samples = Y.shape[0]
    
    for iteration in range (iterations):
        #hipotese
        h = X.dot(theta)
        #diferenca entre hipotese e target
        loss = np.array(h - Y).T
        #Gradiente descendente
        XLoss = X.T.dot(loss.T)
        
        gradient = XLoss/qut_samples
        #changing values of parameters
        theta = theta - (alpha * gradient)
        #New cost value
        cost = cost_function(X,Y,theta)
        cost_history[iteration] = cost
        #print ("Iteration: %d | Cost: %f" % (iteration+1, cost))
    return theta, cost_history



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
    reg = linear_model.LinearRegression()
    reg.fit(X,Y)
    Y_pred = reg.predict(X)
    rmse = np.sqrt(metrics.mean_squared_error(Y, Y_pred))
    return rmse

'''Criando funcao para descobrir o erro root-mean-square 
    input:
        Y: vetor gabarito
        Y_pred: vetor de predicao gerado pelo modelo
    output:
        rmse: erro gerado
'''
def rmse(Y,Y_pred):
    rmse = np.sqrt(((Y - Y_pred) ** 2).mean())
    return float(rmse)



'''Criando funcao para cencontrar valores otimos de theta, usando equacoes normais. 
    input:
        Y: vetor gabarito
        X: matriz de features
        theta: vetor inicial de pesos
    output:
        theta: novos valores para os vetor de pesos
'''
def normal_equation(X,Y,theta):
    theta_ = theta.T
   
    primeiro = X.T.dot(X)
    segundo = np.linalg.pinv(primeiro)
    terceiro = segundo.dot(X.T)
    quarto = terceiro.dot(Y)
    theta = quarto
    return theta