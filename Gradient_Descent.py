import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        temp += 1/m * ( hypo_fn(theta, data_x[i]) - data_y[i] )**2

    return temp

#===============================================================
# Gradient of cost function
#===============================================================
def gradient_cost_fn(hypo_fn, theta, data_xi, data_yi, j):
    return 2.0/m * ( data_yi - hypo_fn(theta, data_xi) )*data_xi[j]

#===============================================================
# Updating theta vetor
#===============================================================
def updating_theta(hypo_fn, alpha, theta, data_x, data_y, j):
    
    right_term = 0
    theta_j = 0

    for i in range(m):
        right_term += gradient_cost_fn(hypo_fn, theta, data_x[i], data_y[i], j)

    theta_j  = theta[j] + alpha*right_term
    
    return theta_j

#===============================================================
# run gradient descent
#===============================================================
def gradient_descent_batch( m, n, theta, alpha, data_x, data_y,
                            error_tolerance, max_iter ):

    converged = False
    current_iter = 0
    value_0 = cost_fn(hypo_fn, theta, data_x, data_y)
    theta_new = np.zeros(n)
    while not converged:
        
        for j in range(n):
            theta_new[j] = updating_theta( hypo_fn, alpha, theta, data_x, data_y, j )

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

    theta_final = gradient_descent_batch( m, n, theta, alpha, data_x, data_y,
                                          error_tolerance, max_iter )


    print (theta_final, id(theta))
    print ('gradient descent complete!')

    plot_result( data_x, data_y, theta_final )
