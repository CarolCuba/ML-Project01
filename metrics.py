import numpy as np

'''Criando funcao para descobrir o erro root-mean-square 
    input:
        Y: vetor gabarito
        Y_pred: vetor de predicao gerado pelo modelo
    output:
        rmse: erro gerado
'''
def rmse(Y,Y_pred):
    rmse = np.sqrt(((Y - Y_pred) ** 2).mean())
    return rmse 

