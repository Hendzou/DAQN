# DAQN
 implementation of the paper Deep Auto-encoder and Q-Network by Daiki Kimura, IBM Research AI
Ojective: implement the method as described in the paper and apply it to a real human game with descrete actions : Rock Paper scissors
Step 1 : pre-train an auto-encoder using 3 hand types images (rock, paper, and scissors) and background images
Step 2 : Use the encoder layers to form a Q-network and train it with Q-learning algorithm
The file auto.py is to train the auto-encoder.
The file rps.py provides the environment of the game, it gives an image of a hand as the state of the game, and take 
