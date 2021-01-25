# Sample input: x_gate.txt

import sys, os, math, random


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
def ff(ts, xv, weights, t_funct):  # this method is basically just forward feeding but you need to maintain the structure
   """ ff coding goes here """
   # print("training set", ts)
   # print("x-values", xv)
   # print("weights", weights)
   counter = 0
   # while counter < len(weights) - 1:
   #    wire = dot_product()
   for x in range(1, len(xv)):  # going through the input values
      if x < len(xv) - 1:  # if we are not on the output layer and we are not done
         wire = dot_product(xv[x-1], weights[x-1], len(xv[x]))  # do the multiplication between value and weight
         evaluation = transfer(t_funct, wire)  # run the function on the wire value
         xv[x] = evaluation  # update the current node value
      else:  # at the output layer and finished
         # print("at the else")
         final_output = dot_product(xv[x-1], weights[x-1],
                                    len(xv[x]))  # we only want to multiply, we don't want to the evaluation
         xv[x] = final_output  # update last node

   err = (ts[-1] - xv[-1][0]) ** 2 / 2  # error calculation with the formula like the sheet

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
         while z < len(ev[x-1]):  # iterating through the the layer before
            layer = x-1  # the current layer
            position = y * len(ev[layer]) + z  # position on the ff network
            if layer != 0:  # you aren't done with the layer nodes yet
               weight = weights[layer][position]  # the weight from the ff network
               part = xv[layer][z] * (1 - xv[layer][z])  # this is the x(1-x) part
               error = ev[x][y]  # the error value from the ff we got
               print(x, z)
               print(ev)
               ev[layer][z] = weight * error * part
            negative_grad[layer][position] = ev[x][y] * xv[layer][z]  # this is the negative gradient, error times x
            z = z + 1  # iterate forward
         y = y + 1  # iterate forward
      x = x - 1  # iterate backwards

   return ev, negative_grad


# update all weights and return the new weights
# Challenge: one line solution is possible
def update_weights(weights, negative_grad, alpha):  # this method is creating the final/complete network like at the end of the worksheet
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


def main():
   file = sys.argv[1]  # only one input (a txt file with training set data)
   if not os.path.isfile(file): exit("Error: training set is not given")
   t_funct = 'T3'  # we default the transfer(activation) function as 1 / (1 + math.e**(-x))
   training_set = [[float(x) for x in line.split() if x != '=>'] for line in open(file, 'r').read().splitlines() if
                   line.strip() != '']
   # print (training_set) #[[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 0.0]]
   layer_counts = [len(training_set[0]), 2, 1, 1]
   print('layer counts', layer_counts)  # This is the first output. [3, 2, 1, 1] with teh given x_gate.txt

   ''' build NN: x nodes and weights '''
   x_vals = [[temp[0:len(temp) - 1]] for temp in training_set]  # x_vals starts with first input values
   # print (x_vals) # [[[1.0, -1.0]], [[-1.0, 1.0]], [[1.0, 1.0]], [[-1.0, -1.0]], [[0.0, 0.0]]]
   # make the x value structure of the NN by putting bias and initial value 0s.
   for i in range(len(training_set)):
      for j in range(len(layer_counts)):
         if j == 0:
            x_vals[i][j].append(1.0)
         else:
            x_vals[i].append([0 for temp in range(layer_counts[j])])
   # print (x_vals) # [[[1.0, -1.0, 1.0], [0, 0], [0], [0]], [[-1.0, 1.0, 1.0], [0, 0], [0], [0]], ...

   # by using the layer counts, set initial weights [3, 2, 1, 1] => 3*2 + 2*1 + 1*1: Total 6, 2, and 1 weights are
   # needed
   weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_counts[i] * layer_counts[i + 1])] for i in
              range(len(layer_counts) - 1)]
   # weights = [[1.35, -1.34, -1.66, -0.55, -0.9, -0.58, -1.0, 1.78], [-1.08, -0.7], [-0.6]]   #Example 2 print (
   # weights)    #[[2.0274715389784507e-05, -3.9375970265443985, 2.4827119599531016, 0.00014994269071843774,
   # -3.6634876683142332, -1.9655046461270405] [-3.7349985848630634, 3.5846029322774617] [2.98900741942973]]

   # build the structure of BP NN: E nodes and negative_gradients
   E_vals = [[*i] for i in x_vals]  # copy elements from x_vals, E_vals has the same structures with x_vals
   negative_grad = [[*i] for i in
                    weights]  # copy elements from weights, negative gradients has the same structures with weights
   errors = [10] * len(training_set)  # Whenever FF is done once, error will be updated. Start with 10 (a big num)
   count = 1  # count how many times you trained the network, this can be used for index calc or for decision making of 'restart'
   alpha = 0.3

   # calculate the initial error sum. After each forward feeding (# of training sets), calculate the error and store at error list
   for k in range(len(training_set)):
      x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
   err = sum(errors)


   while err > .9:  # we will refine further once the error is less than 1
      weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_counts[i] * layer_counts[i + 1])] for i in
              range(len(layer_counts) - 1)]  # reset all the weights
      for k in range(len(training_set)):
         x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)  # forward feed again
      err = sum(errors)  # resum the errors
      # print(err)


   # print("here")
   while err >= .01 and count < 100001:  # the finer tuning
      current_iteration = (count % len(training_set))  # calculate the current iteration
      x_vals[current_iteration], errors[current_iteration] = ff(training_set[current_iteration], x_vals[current_iteration],
                                                                weights, t_funct)  # calculate the current ff results
      if current_iteration == len(training_set) - 1:  # if you have gone through all the training sets
         err = sum(errors)  # recalculate the sum of the errors
         # print(err)
         if err < .25:  # if less than .25
            alpha = 0.02  # reduce the alpha so it is slowed down
         elif count > 2501 and err > .49:  # if it is greater than .5
            count = 1  # reset the count
            alpha = 0.25  # slightly reduce the alpha
            weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_counts[i] * layer_counts[i + 1])] for i
                       in range(len(layer_counts) - 1)]  # reset all the weights
      E_vals[current_iteration], negative_grad = bp(training_set[current_iteration], x_vals[current_iteration], weights,
                                                    E_vals[current_iteration], negative_grad)  # run back propogation
      weights = update_weights(weights, negative_grad, alpha)  # update the weights for the new NN
      count = count + 1  # increase the count


   ''' 
   while err is too big, reset all weights as random values and re-calculate the error sum.

   '''

   ''' 
   while err does not reach to the goal and count is not too big,
      update x_vals and errors by calling ff()
      whenever all training sets are forward fed, 
         check error sum and change alpha or reset weights if it's needed
      update E_vals and negative_grad by calling bp()
      update weights
      count++
   '''
   # print final weights of the working NN
   print('weights:')
   for w in weights: print(w)


if __name__ == '__main__': main()
