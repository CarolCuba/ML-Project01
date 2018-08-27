import numpy as np


'''Criando a funcao de custo
    input:
        X: matriz de features
        Y: matriz de gabarito
        theta: array de pesos
    output:
        custo: custo da iteração
'''
def cost_function(X,Y,theta):
    qut_samples = Y.shape[1]
    h = theta.dot(X)
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
    qut_samples = Y.shape[1]
    
    for iteration in range (iterations):
        #hipotese)
        h = theta.dot(X)
        #diferenca entre hipotese e predicao
        loss = h - Y
        #Gradiente descendente
        X_aux = X.T
        XLoss = loss.dot(X_aux)
        
       
        gradient = XLoss/qut_samples
        #changing values of parameters 
        theta = theta - alpha * gradient
        #New cost value
        cost = cost_function(X,Y,theta)
        cost_history[iteration] = cost
        #print ("Iteration: %d | Cost: %f" % (iteration+1, cost))
    return theta, cost_history

    
    
    

