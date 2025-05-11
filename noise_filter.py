'''the model compression algorithm'''
import numpy as np
from rmt import neural
from sympy import *
import scipy
import pandas as pd
import tensorflow as tf

def get_lambda(zero, alpha):
    solution = []
    for a in alpha:
        for value in a:
            if (value > zero[1] and value < zero[2]) or value > zero[3]:
                solution.append(value)
    return solution

def get_lambda_2(zero, alpha):
    solution = []
    for a in alpha:
        for value in a:
            if value > zero[1]:
                solution.append(value)
    return solution

data_name='mnist'
model_name='MLP'
total_layer=4
for noise in [0.0]:

    NN_1024 = neural.NNModel(path=f'model_{model_name}_{data_name}_noise{noise}',
                                    datasets=neural.get_mnist_fc_std(noise), batch_size=128)
    p=[min(NN_1024.get_weight(layer_index=i).shape) for i in range(total_layer-1)] 
    N=[max(NN_1024.get_weight(layer_index=i).shape) for i in range(total_layer-1)] 
    C=np.array(p)/np.array(N)
    svd=[np.linalg.svd(NN_1024.get_weight(layer_index=i), full_matrices=False) for i in range(total_layer-1)]
    singular_value=[s[1] for s in svd]
    eigenvalue=[s**2 for s in singular_value]
    parameters=scipy.io.loadmat(f'parameters_{model_name}_{data_name}_noise{noise}/our_estimate.mat')
    K=parameters['K'].ravel()
    t1=parameters['t1'].ravel()
    delta1=parameters['delta1'].ravel()
    delta2=parameters['delta2'].ravel()

    eigen_alpha,zero=[],[]

    for i in range(total_layer-1):
        alpha = symbols('alpha',real=True)  # obtain the zero points of the derivative of g(x) from Eq. (10).
        eq = alpha + (p[i] - K[i]) / N[i] * alpha * t1[i] * delta1[i] / (alpha - delta1[i]) + (p[i] - K[i]) / N[
            i] * alpha * (1 - t1[i]) * delta2[i] / (alpha - delta2[i])
        dalpha = diff(eq, alpha)
        eq2 = Eq(dalpha, 0)
        solutions = solve(eq2, alpha)

        Alpha = []
        lbda = symbols('lbda')
        for x in solutions:  # obtain the corresponding boundary points \beta_i from Eq. (12).
            E = Eq(lbda, x + (p[i] - K[i]) / N[i] * x * t1[i] * delta1[i] / (x - delta1[i]) + (p[i] - K[i]) / N[i] * x * (
                        1 - t1[i]) * delta2[i] / (x - delta2[i]))
            Alpha.append(solve(E, lbda)[0])

        eigen_alpha.append(np.array(Alpha).astype(np.float32).tolist())
        where_zero = [np.where(eigenvalue[i] > np.float32(alpha))[0][-1] for alpha in Alpha][::-1]
        zero.append(where_zero)
        Alpha_2 = []
        alpha_2 = symbols('alpha_2')
        if len(where_zero)==4:
            spike_value = np.append(eigenvalue[i][:where_zero[0] + 1], eigenvalue[i][where_zero[1] + 1:where_zero[2]])
            for x in spike_value:
                E = Eq(x, alpha_2 + (p[i] - K[i]) / N[i] * alpha_2 * t1[i] * delta1[i] / (alpha_2 - delta1[i]) +
                    (p[i] - K[i]) / N[i] * alpha_2 * (1 - t1[i]) * delta2[i] / (alpha_2 - delta2[i]))
                Alpha_2.append(list(solveset(E, alpha_2)))
            a = get_lambda(solutions, Alpha_2)
            eigenvalue_copy = eigenvalue[i].copy()
            eigenvalue_copy[:where_zero[0] + 1] = a[:where_zero[0] + 1]
            eigenvalue_copy[where_zero[1] + 1:where_zero[2]] = a[where_zero[0] + 1:]
        else:
            spike_value = eigenvalue[i][:where_zero[0] + 1]
            for x in spike_value:
                E = Eq(x, alpha_2 + (p[i] - K[i]) / N[i] * alpha_2 * t1[i] * delta1[i] / (alpha_2 - delta1[i]) +
                    (p[i] - K[i]) / N[i] * alpha_2 * (1 - t1[i]) * delta2[i] / (alpha_2 - delta2[i]))
                Alpha_2.append(list(solveset(E, alpha_2)))
            a = get_lambda_2(solutions, Alpha_2)
            eigenvalue_copy = eigenvalue[i].copy()
            eigenvalue_copy[:where_zero[0] + 1] = a
        

        ratioRemoved, accuracies, costs=neural.recover_and_remove_svals(NN_1024,layer_indices=[i],
                                                                    svals_shifted=eigenvalue_copy**0.5,dataset_keys=['test'])

    
    

