# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 20:47:34 2017

@author: Shreyash Soni
"""

from numpy import *

def compute_error_for_line_given_points(b, m, points):
    #initialize at zero
    totalError = 0
    #for every point
    for i in range(0, len(points)):
        #get the x-values
        x = points[i, 0]
        #get the y-values
        y = points[i, 1]
        #get the difference, square it, add it to the total
        totalError += (y-(m * x + b)) ** 2
    #get the average
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    #starting points for our gradient
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #direction with respect to the b and m
        #computing partial derivatives of the error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    #update our b and m values using the partial derivatives
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]
    
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    #initial b and m
    b = starting_b
    m = starting_m
    #gradient descent
    for i in range(num_iterations):
        #update b and m with the more accurate b and m by performing this 
        #gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
        if i % 100 == 0:
            print('After {} iterations, b={:0.9f}, m={:0.9f}, error={:.9f}'.format(i, b, m, compute_error_for_line_given_points(b, m, points)))
            
    return [b, m]

    
def run():
    #step-1 collect the data
    points = genfromtxt('data.csv', delimiter = ',')
     #step-2 define the hyperparameters
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    #step-3 train the model
    print('starting gradient descent at b={:0.9f}, m={:0.9f}, error={:0.9f}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print('Running...')
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print('After {} iterations, b={:0.9f}, m={:0.9f}, error={:0.9f}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
     
if __name__ == '__main__':
    run()