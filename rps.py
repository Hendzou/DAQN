#Rock Paper Scissors simulator
import numpy as np
import os
from PIL import Image

data_dir = r"C:\Users\Hend\Documents\DLCV\DLCV_1800809\Dataset_RPS"
X = []
Y = []

for gesture in os.listdir(data_dir) :
    for file in os.listdir(data_dir + '/' + gesture):
            image = Image.open(data_dir  + '/' + gesture + '/' + file).convert("RGB")
            image = image.resize((120, 120), Image.ANTIALIAS)
            x = np.array(image)
            x = x/255
            X.append(x)
            Y.append(str(gesture))

X = np.array(X)

class RPS:
    def __init__(self):
        ran_round = np.random.randint(len(X))
        self.state = X[ran_round]
        self.hidden_state = Y[ran_round]

    def play(self, a):
        a = int(a)
        if self.hidden_state=='Rock':
            y = 0
        elif self.hidden_state=='Paper':
            y = 1
        else:
            y = 2
        reward = a-y
        if (reward==0) :
            return(0)
        elif abs(reward)<2:
            return(reward)
        else:
            return (int(reward*(-0.5)))
