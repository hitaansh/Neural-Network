import sys
import math
import random
import time
import os


# t_funct is symbol of transfer functions: 'T1', 'T2', 'T3', or 'T4'
# input is a list of input (summation) values of the current layer
# returns a list of output values of the current layer


def transfer(t_funct, input):
   if t_funct == 'T3':
      return [1 / (1 + math.e ** -x) for x in input]
   elif t_funct == 'T4':
      return [-1 + 2 / (1 + math.e ** -x) for x in input]
   elif t_funct == 'T2':
      return [x if x > 0 else 0 for x in input]
   else:
      return [x for x in input]


# returns a list of dot_product result. the len of the list == stage
# dot_product([x1, x2, x3], [w11, w21, w31, w12, w22, w32], 2) => [x1*w11 + x2*w21 + x3*w31, x1*w12, x2*w22, x3*w32]
def dot_product(input, weights, stage):
   return [sum([input[x] * weights[x + s * len(input)] for x in range(len(input))]) for s in range(stage)]


# Complete the whole forward feeding for one input(training) set
# return updated x_vals and error of the one forward feeding
def ff(ts, xv, weights,
       t_funct):  # this method is basically just forward feeding but you need to maintain the structure
   """ ff coding goes here """
   # print("training set", ts)
   # print("x-values", xv)
   # print("weights", weights)
   counter = 0
   # while counter < len(weights) - 1:
   #    wire = dot_product()
   for x in range(1, len(xv)):  # going through the input values
      if x < len(xv) - 1:  # if we are not on the output layer and we are not done
         wire = dot_product(xv[x - 1], weights[x - 1], len(xv[x]))  # do the multiplication between value and weight
         evaluation = transfer(t_funct, wire)  # run the function on the wire value
         xv[x] = evaluation  # update the current node value
      else:  # at the output layer and finished
         # print("at the else")
         final_output = dot_product(xv[x - 1], weights[x - 1],
                                    len(xv[x]))  # we only want to multiply, we don't want to the evaluation
         xv[x] = final_output  # update last node

   # print("expected value: ", ts[-1])
   # print("NN OUTPUT: ", xv[-1][0])
   err = (ts[-1] - xv[-1][0]) ** 2 / 2  # error calculation with the formula like the sheet
   # print("error calculation: ", err)

   return xv, err


# Complete the back propagation with one training set and corresponding x_vals and weights
# update E_vals (ev) and negative_grad, and then return those two lists
def bp(ts, xv, weights, ev, negative_grad):  # this method is what we did in the second network for the worksheet
   """ bp coding goes here """
   # print("ev", type(ev))
   # print("ts", type(ts))
   # print("xv", type(xv[-1]))
   # print(ev)
   ev[-1] = [ts[-1] - xv[-1][0]]  # gets the output value and subtracts it from the ideal value from the training set
   x = len(ev) - 1  # we are going to start at the end of the network. this is the last node
   while x > 0:  # iterating through the network backwards
      y = 0  # start from the top of the layer
      while y < len(ev[x]):  # iterate through the layer until you are at the bottom
         z = 0
         while z < len(ev[x - 1]):  # iterating through the the layer before
            layer = x - 1  # the current layer
            position = y * len(ev[layer]) + z  # position on the ff network
            if layer != 0:  # you aren't done with the layer nodes yet
               weight = weights[layer][position]  # the weight from the ff network
               part = xv[layer][z] * (1 - xv[layer][z])  # this is the x(1-x) part
               error = ev[x][y]  # the error value from the ff we got
               # print(x, z)
               # print(ev)
               ev[layer][z] = weight * error * part
            negative_grad[layer][position] = ev[x][y] * xv[layer][z]  # this is the negative gradient, error times x
            z = z + 1  # iterate forward
         y = y + 1  # iterate forward
      x = x - 1  # iterate backwards

   return ev, negative_grad


# update all weights and return the new weights
# Challenge: one line solution is possible
def update_weights(weights, negative_grad,
                   alpha):  # this method is creating the final/complete network like at the end of the worksheet
   """ update weights (modify NN) code goes here """
   new_weights = []  # list of new weights
   x = 0  # start at the 0th position of the weight layers
   while x < len(weights):  # iterate through the weights layers
      y = 0  # start at the 0th weight
      temp_list = []  # list of weights for this layer
      while y < len(weights[x]):  # iterate through the specific weight layer
         new_weight = (alpha * negative_grad[x][y]) + weights[x][y]  # calculate the new weight with formula given
         temp_list.append(new_weight)  # add the weight to the temporary list
         y = y + 1  # iterate forwards in the weight list
      new_weights.append(temp_list)  # add the temporary weight list to the final weights list
      x = x + 1  # iterate forwards in the weight layers
   return new_weights


def make_data(equation):
   r_square = 0
   output_index = 0
   r = 0
   look_up_output = [[1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
   if equation.find('>=') > -1:
      r_square = float(equation[equation.find('=') + 1])
   elif equation.find('<=') > -1:
      r_square = float(equation[equation.find('=') + 1])
      output_index = 1
   elif equation.find('>') > -1:
      r_square = float(equation[equation.find('>') + 1])
      output_index = 2
   else:
      r_square = float(equation[equation.find('<') + 1])
      output_index = 3
   r = math.sqrt(r_square)
   toRet = []
   x_vals = [-1 * r, -1 * r, r, r, 0]
   y_vals = [r, -1 * r, -1 * r, r, 0]
   for i in range(5):
      toRet.append([x_vals[i], y_vals[i], look_up_output[output_index % 2][i]])

   for i in range(690):
      x, y = round(random.uniform(-1.5, 1.5), 2), round(random.uniform(-1.5, 1.5), 2)
      result = x * x + y * y
      if output_index == 0:
         if result >= r_square:
            toRet.append([x, y, 1])
         else:
            toRet.append([x, y, 0])
      elif output_index == 1:
         if result <= r_square:
            toRet.append([x, y, 1])
         else:
            toRet.append([x, y, 0])
      elif output_index == 2:
         if result > r_square:
            toRet.append([x, y, 1])
         else:
            toRet.append([x, y, 0])
      else:
         if result < r_square:
            toRet.append([x, y, 1])
         else:
            toRet.append([x, y, 0])
   # for x in toRet:
   #    print(x)
   return toRet


def main():
   equation = sys.argv[1]
   t_funct = "T3"
   # print("data set")
   training_set = make_data(equation)
   # print(training_set)
   # print("---------------------------")
   print("layer counts")
   layer_counts = [len(training_set[0]), 10, 10, 2, 1, 1]
   print(layer_counts)
   print("---------------------------")
   # print("input values in (x,y)")
   x_vals = [[temp[0:len(temp) - 1]] for temp in training_set]
   # print(x_vals)
   # print("---------------------------")
   for i in range(len(training_set)):
      for j in range(len(layer_counts)):
         if j == 0:
            x_vals[i][j].append(1)
         else:
            x_vals[i].append([0 for temp in range(layer_counts[j])])
   # print("starting weights")
   weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_counts[i] * layer_counts[i + 1])] for i in
              range(len(layer_counts) - 1)]
   # print(weights)
   # print(len(weights[0]))
   # print("---------------------------")
   E_vals = [[*i] for i in x_vals]
   negative_grad = [[*i] for i in
                    weights]
   errors = [10] * len(training_set)
   count = 1
   alpha = 2.5
   for k in range(len(training_set)):
      x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
   err = sum(errors)
   # print("FIRST ERROR: ", err)
   while err > 120:  # we will refine further once the error is less than 1
      # print("here are heavy tuning")
      # print("error at heavy tuning: ", err)
      weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_counts[i] * layer_counts[i + 1])] for i in
                 range(len(layer_counts) - 1)]  # reset all the weights
      for k in range(len(training_set)):
         x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)  # forward feed again
      err = sum(errors)

   t = time.time()
   best_weight = None
   best_err = 99
   count = 0
   while time.time() - t < 99:
      # print("Time: ", time.time()-t)
      current_iteration = (count % len(training_set))
      x_vals[current_iteration], errors[current_iteration] = ff(training_set[current_iteration],
                                                             x_vals[current_iteration],
                                                             weights, t_funct)
      if current_iteration == len(training_set) - 1:
         err = sum(errors)
         # print("Error: ", err, " Best Error: ", best_err)
         if err < best_err:
            best_err = err
            best_weight = weights
         if err < 3:
            alpha = 0.15
         elif err < 6:
            alpha = 1

      E_vals[current_iteration], negative_grad = bp(training_set[current_iteration], x_vals[current_iteration], weights,
                                                    E_vals[current_iteration], negative_grad)
      weights = update_weights(weights, negative_grad, alpha)  # update the weights for the new NN
      count = count + 1

   print("layerCts: ", layer_counts)
   for w in best_weight: print(w)


if __name__ == '__main__': main()
