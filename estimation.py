import numpy as np
from rmt import neural
from sympy import *
import scipy


def part(start,stop,number):# get the \{u_1,...,u_q\}
    r=[]
    dis=stop-start
    for i in range(1,number+1):
        r.append(start+i*dis/(number+1))
    return r
    
def removebike(N,labda,p): # Initially estimate the number of spikes
    w=np.int32(6*N**0.1)
    theta=[np.exp(2*(labda[i]-labda[i+1])) for i in range(len(labda)-1)]
    EDA,EDB=[],[]
    for k in range(1,w+1):
        tilde_theta=1/(p-k-1)*sum(theta[k:])
        EDA.append(-N*(labda[0]-labda[k])+N*(p-k-1)*np.log(tilde_theta)+2*p*k)
        EDB.append(-N*np.log(p)*(labda[0]-labda[k])+N*(p-k-1)*np.log(tilde_theta)+p*k*log(N))
    return EDA,EDB


model_name='MLP'
data_name='mnist'
total_layer=4
for noise in [0.0]:

    NN_1024 = neural.NNModel(path=f'./model_{model_name}_{data_name}_noise{noise}',
                                    datasets=neural.get_mnist_fc_std(noise), batch_size=128)
    
    svd=[np.linalg.svd(NN_1024.get_weight(layer_index=i), full_matrices=False) for i in range(total_layer-1)]
    svd_value=[s[1] for s in svd]
    tzz_value=[s**2 for s in svd_value]
    p=[min(NN_1024.get_weight(layer_index=i).shape) for i in range(total_layer-1)] 
    N=[max(NN_1024.get_weight(layer_index=i).shape) for i in range(total_layer-1)] 
    C=np.array(p)/np.array(N)
    
    result=dict()
    result['K'],result['t1'],result['delta1'],result['delta2']=[],[],[],[] #store parameters K0,t1,delta1,delta2
    density=dict()
    
    for i in range(total_layer-1):
        EDA1, EDB1 = removebike(N[i], tzz_value[i], p[i])
        k_EDA1 = np.argmin(EDA1) + 1
        k_EDB1 = np.argmin(EDB1) + 1
        K=max(k_EDA1,k_EDB1)
        tzz_remove=tzz_value[i][K:]
        l_min=min(tzz_remove)
        l_max=max(tzz_remove)
        if C[i]==1:
            u_value = np.append(part(-10, 0, 20), part(5 * l_max, 10 * l_max, 20))
        else:
            u_value = np.append(np.append(part(-10, 0, 20), part(0, 0.5 * l_min, 20)), part(5 * l_max, 10 * l_max, 20))
        s = np.array([-(1 - (p[i] - K) / N[i]) / u + 1 / N[i] * np.sum(1 / (tzz_remove - u)) for u in u_value])
        def func(s,t1,delta1,delta2):
            c=(p[i]-K)/N[i]
            return np.where((1-t1>0),-1/s+t1*c*delta1/(1+delta1*s)+(1-t1)*c*delta2/(1+delta2*s),-200)
        bounds=([0.0, l_min, l_min], [1.0, l_max, l_max])
        theta, pcov = scipy.optimize.curve_fit(func, s, u_value, 
                                            bounds=bounds, 
                                            maxfev=10000)
        [t1,delta1,delta2]=theta
        result['K'].append(K)
        if t1>0.5:
            result['t1'].append(t1)
            result['delta1'].append(delta1)
            result['delta2'].append(delta2)
        else:
            result['t1'].append(1-t1)
            result['delta1'].append(delta2)
            result['delta2'].append(delta1)

    scipy.io.savemat(f'./parameters_{model_name}_{data_name}_noise{noise}/our_estimate.mat',result)
