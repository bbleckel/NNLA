import emnist as em
import matplotlib.pyplot as plt
import random

def show(img, let):
    plt.imshow(img)
    plt.gray()
    print("Actual label: {}".format(let))
    plt.show()

images, labels = em.extract_training_samples("letters")
letters = {}
for i in range(26):
    letters[i+1] = chr(97+i)

for _ in range(5):
    i = random.randint(0,len(images))
    print(i)
    show(images[i], letters[labels[i]])