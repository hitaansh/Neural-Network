import sys, os, math


# t_funct is symbol of transfer functions: 'T1', 'T2', 'T3', or 'T4'
# input is a list of input (summation) values of the current layer
# returns a list of output values of the current layer
def transfer(t_funct, input):
   if t_funct.upper() == "T1":
      return input
   elif t_funct.upper() == "T2":
      to_return = []
      for x in input:
         if x > 0:
            to_return.append(x)
         else:
            to_return.append(0)
      return to_return
   elif t_funct.upper() == "T3":
      return [(1 / (1 + math.e ** (-1 * x))) for x in input]
   elif t_funct.upper() == "T4":
      return [(-1 + (2 / (1 + math.e ** (-1 * x)))) for x in input]


# example: 4 inputs, 12 weights, and 3 stages(the number of next layer nodes)
# weights are listed like Example Set 1 #4 or on the NN_Lab1_description note
# returns a list of dot_product result. the len of the list == stage
# Challenge? one line solution is possible
def dot_product(input, weights, stage):
   # print(len(weights))
   stage = len(weights) // len(input)
   # print(stage)
   next_layer_results = []
   for s in range(stage):
      node_sum = 0
      for w in range(len(input)):
         specific_weight = weights[w + s * len(input)]
         node_sum = node_sum + (input[w] * specific_weight)
      next_layer_results.append(node_sum)

   return next_layer_results


# file has weights information. Read file and store weights in a list or a nested list
# input_vals is a list which includes input values from terminal
# t_funct is a string, e.g. 'T1'
# evaluate the whole network (complete the whole forward feeding)
# and return a list of output(s)
def evaluate(file, input_vals, t_funct):
   # print(file, input_vals, t_funct)
   print("input_vals", input_vals)
   w_file = open(file)
   all_weights = []
   for line in w_file.readlines():
      weights = line.split(" ")
      temp = []
      for weight in weights:
         temp.append(float(weight))
      all_weights.append(temp)
   counter = 0
   # print('made it to after weights')
   while counter < (len(all_weights) - 1):
      result = dot_product(input_vals, all_weights[counter], counter)
      # print(result)
      input_vals = transfer(t_funct, result)
      counter += 1

   # print('made it to after weight multiplication')
   # print(input_vals)
   # print(all_weights[counter])
   output = []
   for i in range(len(input_vals)):
      output.append(input_vals[i] * all_weights[counter][i])
   #print(output)



   # output = dot_product(input_vals, all_weights[counter], counter)
   # print(output)
   return output


def main():
   args = sys.argv[1:]
   file, inputs, t_funct, transfer_found = '', [], 'T1', False
   for arg in args:
      if os.path.isfile(arg):
         file = arg
      elif not transfer_found:
         t_funct, transfer_found = arg, True
      else:
         inputs.append(float(arg))
   if len(file) == 0: exit("Error: Weights file is not given")
   li = (evaluate(file, inputs, t_funct))
   for x in li:
      print(x, end=' ')


if __name__ == '__main__': main()
