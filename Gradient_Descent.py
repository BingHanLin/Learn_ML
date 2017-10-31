import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

'''
A code for gradient descent method.
https://github.com/mattnedrich/GradientDescentExample
'''

#===============================================================
# Hypothesis function
#===============================================================
def hypo_fn(theta, data_xi):
    temp = 0
    data = zip(theta, data_xi)

    for theta_i, x_i in data: 
        temp += theta_i*x_i

    return temp

#===============================================================
# Cost function
#===============================================================
def cost_fn(hypo_fn, theta, data_x, data_y):
    temp = 0

    for i in range(m):
        temp += 1.0/(2.0*m) * ( hypo_fn(theta, data_x[i]) - data_y[i] )**2

    return temp

#===============================================================
# Gradient of cost function
#===============================================================
def gradient_cost_fn(hypo_fn, theta, data_xi, data_yi, num_m, j ):
    return 1.0/num_m * ( hypo_fn(theta, data_xi) - data_yi )*data_xi[j]

#===============================================================
# run batch gradient descent
#===============================================================
def batch_GD( m, n, theta, alpha, data_x, data_y, error_tolerance, max_iter ):

    converged = False
    current_iter = 0
    value_0 = cost_fn(hypo_fn, theta, data_x, data_y)
    theta_new = np.zeros(n)
    while not converged:
        
        for j in range(n):

            right_term = 0
            num_m = m

            for i in range(m):
                right_term += -gradient_cost_fn( hypo_fn, theta, data_x[i], data_y[i], num_m, j )

            theta_new[j]  = theta[j] + alpha*right_term


        theta = theta_new

        value = cost_fn(hypo_fn, theta, data_x, data_y)

        if current_iter > max_iter:
            converged = True

        elif ( abs(value-value_0) <= error_tolerance ):
            converged = True

        current_iter += 1
        value_0  = value
        
        print ("Iterations: {0}, \nValue of cost function: {1}\n".format(current_iter, value))

    return theta

#===============================================================
# run stochastic(or mini batch) gradient descent
#===============================================================
def stochastic_GD( m, n, theta, alpha, data_x, data_y, error_tolerance, max_iter, num_batch=1 ):

    converged = False
    current_iter = 0
    value_0 = cost_fn(hypo_fn, theta, data_x, data_y)
    theta_new = np.zeros(n)
    while not converged:
        
        for j in range(n):

            right_term = 0
            num_m = num_batch
            ind = []
            for _ in range(num_m):
                ind.append( random.randint(0, m - 1) )  #take random integer from 0 ~ m

            for i in range(num_m):
                right_term += -gradient_cost_fn( hypo_fn, theta, data_x[ind[i]], data_y[ind[i]], num_m, j  )

            theta_new[j]  = theta[j] + alpha*right_term


        theta = theta_new

        value = cost_fn(hypo_fn, theta, data_x, data_y)

        if current_iter > max_iter:
            converged = True

        elif ( abs(value-value_0) <= error_tolerance ):
            converged = True

        current_iter += 1
        value_0  = value
        
        print ("Iterations: {0}, \nValue of cost function: {1}\n".format(current_iter, value))

    return theta


#===============================================================
# plot result
#===============================================================
def plot_result( data_x, data_y, theta ):
    
    x_reg = np.ones((2,n))
    y_reg = np.zeros(2)

    x_reg[0,1] = data_x[:,1].min()
    x_reg[1,1] = data_x[:,1].max()


    y_reg[0] = hypo_fn(theta, x_reg[0,:])
    y_reg[1] = hypo_fn(theta, x_reg[1,:])

    plt.plot(data_x[:,1], data_y,'ro')
    plt.plot(x_reg[:,1], y_reg, 'b', LineWidth = 3)
    plt.show()
#===============================================================
# Example
#===============================================================
if __name__ == '__main__':

    df = pd.read_csv('data.csv')        # read traingin data

    df.insert(0,'X0',1)                 # add frictional feature

    # separate x, y data and turn into numpy
    data_x = df.drop('FINAL', axis=1).as_matrix()
    data_y = df['FINAL'].as_matrix()


    m = data_x.shape[0]                 # the number of training examples
    n = data_x.shape[1]                 # the number of features(include frictional feature)
    theta = np.zeros(n)                 # initial guess of the parameters 
    alpha = 0.0001                      # learning rate

    error_tolerance = 10e-5             # tolerance value of error
    max_iter = 10000                    # maximum number of iterations

    # theta_final = batch_GD( m, n, theta, alpha, data_x, data_y,
    #                         error_tolerance, max_iter )

    theta_final = stochastic_GD( m, n, theta, alpha, data_x, data_y,
                                 error_tolerance, max_iter, num_batch=5 )

    print (theta_final, id(theta))
    print ('gradient descent complete!')

    plot_result( data_x, data_y, theta_final )
