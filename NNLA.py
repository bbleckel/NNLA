import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt
import emnist as em

SIGMOID = lambda x: 1/(1+math.e**(-x))
LR = 1

ACC = []

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

def main():
    epochs = 5
    # train_data = []
    # read_data("optdigits-32x32.tra", train_data)

    # test_data = []
    # read_data("optdigits-32x32.tes", test_data)
    train, train_labels = em.extract_training_samples("digits")
    test, test_labels = em.extract_test_samples("digits")
    train_data = []
    for i,img in enumerate(train):
        train_data.append(Image(img, train_labels[i]))

    test_data = []
    for i,img in enumerate(test):
        train_data.append(Image(img, test_labels[i]))

    weights = np.random.rand(10,1025) * 2 - 1

    #train
    for e in range(epochs):
        count = 0
        total = 0
        np.random.shuffle(train_data)
        for img in train_data:

            data = np.array(img.flatten(bias=True))

            output = SIGMOID(weights.dot(data))

            err = img.value_arr - output

            if np.argmax(output) == img.value:
                count += 1
            total += 1

            # backprop
            # weights += data.dot(err * LR * (output * (1-output)))
            for i in range(len(data)):
                weights[:,i] += (err.dot(data[i]).dot(output.dot(1-output)) * LR)
            progressBar(e+1, count, len(train_data), 100*count/total)
        ACC.append(count/total)
        progressBar(e+1, count, len(train_data), 100*count/total, end=True)

    #save model
    outfile = "model.nn"
    with open(outfile, "w") as file:
        for r in weights:
            file.write(",".join([str(i) for i in r]))
            file.write("\n")
        file.close()

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
    # img = test_data[-10]
    # output = list(img.output.argsort())
    # output.reverse()
    # print(img)
    # print(img.value, output)

    # plt.plot(ACC)
    # plt.show()

def progressBar(epoch, value, endvalue, acc, end=False, bar_length=40):

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rEpoch {0}: [{1}] acc={2:.0f}%    ".format(epoch, arrow + spaces, acc))
    sys.stdout.flush()
    if end:
        sys.stdout.write("\rEpoch {0}: [{1}] acc={2:.2f}%    \n".format(epoch, arrow + spaces, acc))
        sys.stdout.flush()

main()