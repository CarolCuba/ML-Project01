import numpy as np




'''Criando funcao para cencontrar valores otimos de theta, usando equacoes normais. 
    input:
        Y: vetor gabarito
        X: matriz de features
        theta: vetor inicial de pesos
    output:
        theta: novos valores para os vetor de pesos
'''
def normalize_equation(X,Y,theta):
    
    X_transpose = X
    X_ = X.T
    Y_ = Y.T
    theta_ = theta.T
   
    primeiro = X_transpose.dot(X_)
    segundo = np.linalg.pinv(primeiro)
    terceiro = segundo.dot(X_transpose)
    quarto = terceiro.dot(Y_)
    theta = quarto
    return theta
   
