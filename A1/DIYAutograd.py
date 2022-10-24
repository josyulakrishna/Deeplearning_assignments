import itertools
import pdb

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


def main():

  functionUnitTests()
  backpropUnitTests()



  #######################
  # Gradient Ascent Demo
  #######################

  x1 = Variable(1, name="x1")
  x2 = Variable(-1/2, name="x2")


  for i in range(500):
    y = (x1+x2)*(x1*x2 + x1*(x2**2))
    y.backward()
    x1.value += 0.1*x1.grad
    x2.value += 0.1*x2.grad
    x1.grad = x2.grad = 0

  #plt.figure(figsize=(15,7))
  #y.drawGraph()
  #plt.show()

  print(x1,x2, y)



class Variable:

  id_iter = itertools.count() # Assign unique ID to each variable

  def __init__(self, value, arg_nodes=[], func=None, name=None):
    self.value = value            # Output value
    self.children = arg_nodes     # Input nodes (None if constant)
    self.func = func              # Function mapping input to output (None if constant)
    self.name = name              # Optional name

    self.grad = 0                 # Derivative during backprop
    self.id = next(Variable.id_iter) # Unique ID



  def backward(self):

    # Compute a topological ordering so each node is processed
    # after all parents have sent gradients to it
    # Run backprop following the topological sort
    nodes = self.topoSort()
    self.grad = 1
    print(nodes)
    for node in nodes:
      dy_dp=node.grad
      if node.func:
        dp_di = node.func.backward(node, node.children)
        for i,c in enumerate(node.children):
            c.grad += dy_dp*dp_di[i]

  def nodeName(self):
    if self.func != None:
      return self.func.label
    if self.name != None:
      return self.name
    return str(np.round(self.value,3))

  def __repr__(self):
    rep = "Variable("+str(self.value)
    if self.name:
      rep += ", "+self.name
    rep += ")"
    return rep

  def drawGraph(self):

    G = nx.DiGraph(directed=True)

    stack = [self]
    visited = set()
    first = True

    G.add_node(self.id, txtlabel=self.nodeName(), terminal= 2)
    visited.add(self.id)

    while len(stack) > 0:
      active = stack.pop()
      stack.extend(active.children)

      for c in active.children:
        if c not in visited:
          visited.add(c.id)
          G.add_node(c.id, txtlabel=c.nodeName(), terminal= c.func == None)
        G.add_edge(c.id, active.id, label=str(np.round(c.value,3)))

    # Draw computation graph
    edge_labels = nx.get_edge_attributes(G,'label')
    node_labels = nx.get_node_attributes(G, 'txtlabel')
    terminal_labels = list(nx.get_node_attributes(G, 'terminal').values())
    pos = nx.nx_pydot.graphviz_layout(G, root=self.id, prog="dot")
    pos = {node: (-y,x) for (node, (x,y)) in pos.items()}

    nx.draw(G, pos=pos, arrows=True, with_labels = False, node_size=2000, node_color=terminal_labels, cmap="Set2")
    nx.draw_networkx_labels(G, pos=pos, labels = node_labels)
    nx.draw_networkx_edge_labels(G,pos=pos,edge_labels=edge_labels)

  # Produces an topological ordering where node i comes before node j if and only if no edge j,i exists in the graph
  # Uses Kahn's algorithm
  def topoSort(self):

    # Collect degree information
    stack = [self]
    in_degree = defaultdict(lambda : 0)
    visited = set()

    while stack:
      active = stack.pop()
      if active.id in visited:
        continue

      visited.add(active.id)

      stack.extend(active.children)
      for c in active.children:
        in_degree[c.id] += 1


    # Iteratively identify nodes with degree zero
    topo_order = []
    nodes_with_degree_zero =  [self]  # Assume self is only node with degree zero -- true in backprop.
    while nodes_with_degree_zero:
      active = nodes_with_degree_zero.pop()
      topo_order.append(active)

      # Pretend this node is gone, what nodes now have degree zero
      for c in active.children:
        in_degree[c.id] -= 1

        if in_degree[c.id] == 0:
          nodes_with_degree_zero.append(c)
    return topo_order


  # Overwriting basic operations
  def __add__(self, other):
    return Add.forward(self,other)

  def __sub__(self,other):
    return Add.forward(self, Prod.forward(other,-1))

  def __mul__(self, other):
    return Prod.forward(self,other)

  def __pow__(self, other):
    return Pow.forward(self, other)

  def __truediv__(self, other):
    return Prod.forward(self, Inv.forward(other))

  def convert(a):
    #If not a Variable, assume a constant and try to convert to one.
    if not isinstance(a, Variable):
      return Variable(a, name="const")
    return a


###################################
# Q5 Implement Unfinished Functions
###################################

# Forward: a - > 1/a
class Inv:
  label = "inv"

  def forward(a):
    a = Variable.convert(a)
    return Variable(1 / a.value, [a], Inv)

  def backward(parent, children):
    return [-1 / (children[0].value ** 2)]


# Forward: a,b -> a+b
class Add:
  label = "+"

  def forward(a, b):
    a = Variable.convert(a)
    b = Variable.convert(b)
    return Variable(a.value + b.value, [a, b], Add)

  def backward(parent, children):
    return [1, 1]


# Forward: a,b -> a*b
class Prod:
  label = "*"

  def forward(a, b):
    a = Variable.convert(a)
    b = Variable.convert(b)
    return Variable(a.value * b.value, [a, b], Prod)

  def backward(parent, children):
    return [children[1].value, children[0].value]


# Forward: a -> ln(a)
class Ln:
  label = "ln"

  def forward(a):
    a = Variable.convert(a)
    return Variable(np.log(a.value), [a], Ln)

  def backward(parent, children):
    return [1 / children[0].value]


# Forward: a,b -> a^b
class Pow:
  label = "pow"

  def forward(a, b):
    a = Variable.convert(a)
    b = Variable.convert(b)
    return Variable(a.value ** b.value, [a, b], Pow)

  def backward(parent, children):
    return [children[1].value * ((children[0].value) ** ((children[1].value) - 1)),
            ((children[0].value ** children[1].value) * np.log(children[0].value))]


######################################
# Trig Functions
######################################

# Forward: a -> sin(a)
class Sin:
  label = "sin"

  def forward(a):
    a = Variable.convert(a)
    return Variable(np.sin(a.value), [a], Sin)

  def backward(parent, children):
    return [np.cos(children[0].value)]


# Forward: a -> cos(a)
class Cos:
  label = "cos"

  def forward(a):
    a = Variable.convert(a)
    return Variable(np.cos(a.value), [a], Cos)

  def backward(parent, children):
    return [-np.sin(children[0].value)]


# Forward: a -> tan(a)
class Tan:
  label = "tan"

  def forward(a):
    a = Variable.convert(a)
    return Variable(np.tan(a.value), [a], Tan)

  def backward(parent, children):
    return [1 / np.cos(children[0].value) ** 2]


######################################
# Max/Min Functions
######################################

# Forward: a,b -> max(a,b)
class Max:
  label = "max"

  def forward(a, b):
    a = Variable.convert(a)
    b = Variable.convert(b)
    return Variable(np.max([a.value, b.value]), [a, b], Max)

  def backward(parent, children):
    return [1, 0] if children[0].value > children[1].value else [0, 1]

  # Forward: a,b -> min(a,b)


class Min:
  label = "min"

  def forward(a, b):
    a = Variable.convert(a)
    b = Variable.convert(b)
    return Variable(a.value if a.value < b.value else b.value, [a, b], Min)

  def backward(parent, children):
    return [1, 0] if children[0].value < children[1].value else [0, 1]


######################################
# Aliases to make coding cleaner
######################################
def ln(a):  return Ln.forward(a)


def sin(a):  return Sin.forward(a)


def cos(a):  return Cos.forward(a)


def tan(a):  return Tan.forward(a)


def max(a, b):  return Max.forward(a, b)


def min(a, b):  return Min.forward(a, b)


def functionUnitTests():
  # Unit Tests for Functions

  # Inverse
  assert (Inv.forward(Variable(2)).value == 1 / 2)
  assert (Inv.forward(Variable(2)).func == Inv)
  assert (Inv.backward(Variable(1 / 2), [Variable(2)]) == [-0.25])

  # Addition
  assert (Add.forward(Variable(2), Variable(1)).value == 3)
  assert (Add.forward(Variable(2), Variable(1)).func == Add)
  assert (Add.backward(Variable(3), [Variable(2), Variable(1)]) == [1, 1])

  # Natural Log
  assert (abs(Ln.forward(Variable(2)).value - 0.6931471805599453) < 0.001)
  assert (Ln.forward(Variable(2)).func == Ln)
  assert (Ln.backward(Variable(0.6931471805599453), [Variable(2)]) == [0.5])

  # Exponential

  assert (Pow.forward(Variable(2), Variable(4)).value == 16)
  assert (Pow.forward(Variable(2), Variable(4)).func == Pow)
  assert (
    np.all(np.array(Pow.backward(Variable(16), [Variable(2), Variable(4)])) == np.array([32, 11.090354888959125])))

  # Sin
  assert (Sin.forward(Variable(2)).value == np.sin(2))
  assert (Sin.forward(Variable(2)).func == Sin)
  assert (abs(Sin.backward(Variable(np.sin(2)), [Variable(2)])[0] + 0.4161468365471424) < 0.001)

  # Cos
  assert (Cos.forward(Variable(2)).value == np.cos(2))
  assert (Cos.forward(Variable(2)).func == Cos)
  assert (abs(Cos.backward(Variable(np.cos(2)), [Variable(2)])[0] + 0.9092974268256817) < 0.001)

  # Tan
  assert (Tan.forward(Variable(2)).value == np.tan(2))
  assert (Tan.forward(Variable(2)).func == Tan)
  assert (abs(Tan.backward(Variable(np.tan(2)), [Variable(2)])[0] - 5.774399204041917) < 0.001)

  # Min
  assert (Min.forward(Variable(2), Variable(4)).value == 2)
  assert (Min.forward(Variable(2), Variable(4)).func == Min)
  assert (np.all(np.array(Min.backward(Variable(2), [Variable(2), Variable(4)])) == np.array([1, 0])))

  # Max
  assert (Max.forward(Variable(2), Variable(4)).value == 4)
  assert (Max.forward(Variable(2), Variable(4)).func == Max)
  assert (np.all(np.array(Max.backward(Variable(4), [Variable(2), Variable(4)])) == np.array([0, 1])))


def backpropUnitTests():
  # Unit Test 0
  x1 = Variable(5.0, name="x_1")
  a = x1 + 3
  b = a * 3
  c = b / 3
  d = c - 1

  # If you want to display the graph, uncomment these lines.
  # plt.figure(figsize=(18,6))
  # d.drawGraph()
  # plt.show()

  d.backward()
  assert (x1.grad == 1)

  # Unit Test 1

  # Produce example graph from the assignment
  x1 = Variable(5.0, name="x_1")
  x2 = Variable(-2.0, name="x_2")
  x3 = Variable(-5, name="x_3")

  y = ln(x1 + sin(x1)) * max(x2, x3) * 2

  # If you want to display the graph, uncomment these lines.
  # plt.figure(figsize=(18,6))
  # y.drawGraph()
  # plt.show()

  # Execute backpropagation
  y.backward()

  assert (abs(x1.grad - -1.27) < 0.01)
  assert (abs(x2.grad - 2.79) < 0.01)
  assert (abs(x3.grad - 0) < 0.01)

  # Unit Test 2

  # Produce example graph from the assignment
  x1 = Variable(5.0, name="x_1")
  y = Variable(0)

  for i in range(5):
    if i % 2 == 0:
      y = y + x1 * i
    else:
      y = y - x1 / i

      # If you want to display the graph, uncomment these lines.
  # plt.figure(figsize=(18,6))
  # y.drawGraph()
  # plt.show()

  # Execute backpropagation
  y.backward()

  assert (abs(x1.grad - 4.666666666666666) < 0.01)

  # Unit Test 3

  # Produce example graph from the assignment
  x1 = Variable(0.5, name="x_1")
  x2 = Variable(1.0, name="x_2")

  a = sin(x1) - tan(x1) * 50
  b = ln(x1 * x2) + x1 ** x2 + cos(x2) + x2
  c = min(a, b) + max(a, b)

  # If you want to display the graph, uncomment these lines.
  # plt.figure(figsize=(18,6))
  # c.drawGraph()
  # plt.show()

  # Execute backpropagation
  c.backward()
  assert (abs(x1.grad - -61.04473795858586) < 0.01)
  assert (abs(x2.grad - 0.8119554249121308) < 0.01)


if __name__ == "__main__":
  main()
