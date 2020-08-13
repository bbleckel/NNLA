import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt

SIGMOID = lambda x: 1/(1+math.e**(-x))

class Image():
    def __init__(self, data, value):
        self.data = data
        self.value = value
        self.value_arr = np.zeros(10)
        self.value_arr[value] = 1
        self.output = None

    def flatten(self, bias=False):
        flat_image = []
        for x in self.data:
            flat_image.extend(x)
        if bias: flat_image.append(1)
        return flat_image

    def __str__(self):
        out = ""
        for r in self.data:
            for c in r:
                out += str(c)
            out += "\n"
        return out

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

def load_weights(filepath):
    weights = []
    with open(filepath, "r") as file:
        for line in file:
            row = []
            line = line.strip().split(",")
            for w in line:
                row.append(float(w))
            weights.append(np.array(row))
    return np.array(weights)

def main():
    test_data = []
    read_data("optdigits-32x32.tes", test_data)
    print("data loaded")
    weights = load_weights("model.nn")
    print("weights loaded")
    #test
    count = 0
    wrong = {}
    for img in test_data:
        data = np.array(img.flatten(bias=True))

        output = SIGMOID(weights.dot(data))
        img.output = output

        if np.argmax(output) == img.value:
            count += 1
        else:
            wrong[img] = output
    print("Test data acc={:.2f}%".format(100*count/len(test_data)))

    # for x in wrong:
    #     out = wrong[x]
    #     out = list(out.argsort())
    #     out.reverse()
    #     print(x)
    #     print(x.value, out)
    val = 0
    while val != -1:
        val = int(input("Enter an image to check (values in range 0-{}) (-1 to quit)".format(len(test_data)-1)))
        if val < len(test_data) and val >= 0:
            img = test_data[val]
            output = list(img.output.argsort())
            output.reverse()
            print(img)
            print(img.value, output)


main()