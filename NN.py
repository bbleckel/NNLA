# NN tester
import numpy as np
import random
import math
import sys

#CONSTANTS
RAND_MIN = -.1
RAND_MAX = .1
LEARNING_RATE = .1
BATCH_SIZE = 1
SIGMOID_ACTIVATION = lambda x: 1/(1+math.e**(-x))

class Image():
    def __init__(self, data, value):
        self.data = data
        self.value = value

    def flatten(self):
        flat_image = []
        for x in self.data:
            flat_image.extend(x)
        return flat_image

    def __str__(self):
        out = ""
        for r in self.data:
            for c in r:
                out += str(c)
            out += "\n"
        return out

class Node():
    def __init__(self, prev_layer=None, activation=-1, bias=False):
        self.prev_layer = prev_layer
        if prev_layer != None:
            self.incoming_weights = [random.uniform(RAND_MIN, RAND_MAX) for _ in range(len(prev_layer))]
        else:
            self.incoming_weights = None
        self.activation = activation
        self.bias = False

    def set_activation(self, val):
        self.activation = val
        return val

    def calc_activation(self, act_func=SIGMOID_ACTIVATION):
        sum_input = np.sum([self.prev_layer[i].activation*self.incoming_weights[i] for i in range(len(self.prev_layer))])
        self.activation = act_func(sum_input)
        return self.activation

    def backprop(self, err):
        deriv = self.activation * (1-self.activation)
        for i,node in enumerate(self.prev_layer):
            self.incoming_weights[i] += (err * node.activation * deriv * LEARNING_RATE)


    def is_bias_node(self):
        return self.bias

    def __str__(self):
        return str(self.activation)

class Network():
    def __init__(self):
        self.net = []
    
    def add(self, num_nodes, bias=False):
        if len(self.net) > 0:
            prev_layer = self.net[-1]
        else:
            prev_layer = None

        # add all nodes
        x = [Node(prev_layer=prev_layer) for _ in range(num_nodes)]
        # add bias node if not last layer
        if bias: x.append(Node(prev_layer=prev_layer, activation=1, bias=True))

        self.net.append(x)

    def train(self, d_set, epochs=10):
        for epoch in range(epochs):
            correct = 0
            counter = 0
            err = [0]*len(self.net[-1])
            for d in d_set:
                # show data to network
                d_flat = d.flatten()
                for i, val in enumerate(d_flat):
                    self.net[0][i].set_activation(val)
                # process data through network
                max_act = SIGMOID_ACTIVATION(-500)
                max_j = -1
                for j,node in enumerate(self.net[-1]):
                    act = node.calc_activation(SIGMOID_ACTIVATION)
                    if act > max_act:
                        max_act = act
                        max_j = j
                    err[j] += (1 if j == d.value else 0) - act
                    if counter % BATCH_SIZE == 0 :
                        node.backprop(err[j]/BATCH_SIZE)
                        err[j] = 0
                    # node.backprop((1 if j == d.value else 0) - act)
                if max_j == d.value:
                    # print(d)
                    # print(d.value, max_j)
                    # print(["{:.2f}".format(i.activation) for i in self.net[-1]])
                    correct += 1
                counter += 1
                progressBar(epoch+1, counter, len(d_set), 100*correct/counter)

            sys.stdout.write("\rEpoch {}: acc={:.2f}%{}\n".format(epoch+1, 100*correct/len(d_set), " "*50))

    def test(self, test_set):
        correct = 0
        for img in test_set:
            img_flat = img.flatten()
            for i, val in enumerate(img_flat):
                self.net[0][i].set_activation(val)
            # process data through network
            max_act = SIGMOID_ACTIVATION(-500)
            max_j = -1
            for j,node in enumerate(self.net[-1]):
                act = node.calc_activation(SIGMOID_ACTIVATION)
                if act > max_act:
                    max_act = act
                    max_j = j
            if max_j == img.value:
                correct += 1

        print("Testing acc={:.2f}%".format(100*correct/len(test_set)))

    def __str__(self):
        return ' '.join([str(len(i)) for i in self.net])

def read_data(filepath, data_path):
    with open(filepath, 'r') as file:
        start = False
        img = []
        val = -1
        for line in file:
            row = []
            if not start:
                if line[0] == '0':
                    start = True

            if start and line[0] != " ":
                for pix in line.strip():
                    # print(row)
                    row.append(int(pix.strip()))
                img.append(row)

            if line[0] == " ":
                val = int(line.strip())
                data_path.append(Image(img, val))
                start = False
                img = []
                val = -1

def build_network(layers=[]):
    net = Network()
    for l in layers[:-1]:
        net.add(l, bias=True)
    net.add(layers[-1])

    return net

def progressBar(epoch, value, endvalue, acc, bar_length=40):

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rEpoch {0}: [{1}] acc={2:.2f}%    ".format(epoch, arrow + spaces, acc))
    sys.stdout.flush()

def main():
    train_data = []
    read_data("optdigits-32x32.tra", train_data)

    random.shuffle(train_data)

    test_data = []
    read_data("optdigits-32x32.tes", test_data)
    
    net = build_network(layers=[1024, 10])
    print("Layers: {}, Min: {}, Max: {}, LR: {}, BS: {}".format(net, RAND_MIN, RAND_MAX, LEARNING_RATE, BATCH_SIZE))

    net.train(train_data, 5)
    net.test(test_data)

main()