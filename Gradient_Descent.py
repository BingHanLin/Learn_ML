import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
https://github.com/mattnedrich/GradientDescentExample
'''
# hypothesis function
def hypo_fn(theta, data_xi):
    temp = 0

    data = zip(theta, data_xi)

    for theta_i, x_i in data: 
        temp += theta_i*x_i

    return temp

# cost function
def cost_fn(hypo_fn, theta, data_x, data_y):
    temp = 0
    for i in range(m):
        temp += 1/m * ( hypo_fn(theta, data_x[i]) - data_y[i] )**2

    return temp

# gradient of cost function
def gradient_cost_fn(hypo_fn, theta, data_xi, data_yi, j):
    return 2.0/m * ( data_yi - hypo_fn(theta, data_xi) )*data_xi[j]

# updating theta vetor
def updating_theta(hypo_fn, alpha, theta, data_x, data_y, j):
    
    right_term = 0
    theta_j = 0

    for i in range(m):
        right_term += gradient_cost_fn(hypo_fn, theta, data_x[i], data_y[i], j)

    theta_j  = theta[j] + alpha*right_term
    
    return theta_j

# run regression
def run():
    converged = False
    current_iter = 0
    value_0 = cost_fn(hypo_fn, theta, data_x, data_y)

    while not converged:
        
        for j in range(n):
            theta[j] = updating_theta( hypo_fn, alpha, theta, data_x, data_y, j )

        value = cost_fn(hypo_fn, theta, data_x, data_y)

        if current_iter > max_iter:
            converged = True
        elif ( abs(value-value_0) <= error_tolerance ):
            converged = True

        current_iter += 1
        value_0  = value

        print (current_iter,' ',value)
    
    return theta

#===============================================================
# main program
#===============================================================
if __name__ == '__main__':

    df = pd.read_csv('data.csv')   # read traingin data

    df.insert(0,'X0',1)             # add frictional feature

    # separate x, y data and turn into numpy
    data_x = df.drop('FINAL', axis=1).as_matrix()
    data_y = df['FINAL'].as_matrix()

    m = data_x.shape[0]             # the number of training examples
    n = data_x.shape[1]             # the number of features(include frictional feature)
    theta = np.zeros(n)             # initial guess of the parameters 
    alpha = 0.0001                  # learning rate

    error_tolerance = 10e-5         # tolerance value of error
    max_iter = 10000                # maximum number of iterations
    
    run()

    print (theta)
    print ('regression complete!')
    plt.plot(data_x[:,1], data_y,'o')
    # plt.show()