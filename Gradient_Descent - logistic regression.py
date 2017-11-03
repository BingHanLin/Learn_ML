import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

'''
A code for gradient descent method.
https://github.com/mattnedrich/GradientDescentExample
http://terrence.logdown.com/posts/440690-python-super-simple-implementation-of-logistic-regression-classification-and-examples
'''

#===============================================================
# Hypothesis function
#===============================================================
def hypo_fn(theta, data_xi):

    temp = theta.T.dot(data_xi)

    return 1 / (1 + np.exp(-temp))

#===============================================================
# Cost function
#===============================================================
def cost_fn(hypo_fn, theta, data_x, data_y):
    temp = 0

    for i in range(m):
        h = hypo_fn(theta, data_x[i])
        temp += 1.0/ m * ( -data_y[i]*np.log(h) - (1 - data_y[i])*np.log(1-h) )

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
    


    l = np.linspace(-2,2)
    a,b = -theta[1]/theta[2], -theta[0]/theta[2]
    plt.plot(l, a*l + b, 'k--')

    data = pd.concat([data_x, data_y], axis=1)

    data1 = data[data.ix[:, -1]==0]
    data2 = data[data.ix[:, -1]==1]
    
    plt.plot(data1.ix[:,1], data1.ix[:,2],'ro')
    plt.plot(data2.ix[:,1], data2.ix[:,2],'bo')

    # plt.plot(x_reg[:,1], y_reg, 'b', LineWidth = 3)
    
    plt.show()
#===============================================================
# Example
#===============================================================
if __name__ == '__main__':

    df = pd.read_csv('data - logistic regression.csv')        # read traingin data

    df.insert(0,'X0',1)                 # add frictional feature

    # separate x, y data and turn into numpy
    data_x = df.drop('RESULT', axis=1).as_matrix()
    data_y = df['RESULT'].as_matrix()


    m = data_x.shape[0]                 # the number of training examples
    n = data_x.shape[1]                 # the number of features(include frictional feature)
    theta = np.zeros(n)                 # initial guess of the parameters 
    alpha = 0.1                      # learning rate

    error_tolerance = 10e-5             # tolerance value of error
    max_iter = 1000                    # maximum number of iterations

    # theta_final = batch_GD( m, n, theta, alpha, data_x, data_y,
    #                         error_tolerance, max_iter )

    theta_final = stochastic_GD( m, n, theta, alpha, data_x, data_y,
                                 error_tolerance, max_iter, num_batch=5 )

    print (theta_final)
    print ('gradient descent complete!')

    plot_result( df.drop('RESULT', axis=1), df['RESULT'], theta_final )
