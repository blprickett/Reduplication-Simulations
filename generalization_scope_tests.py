import numpy as np
import seq2seq
from seq2seq.models import Seq2Seq
from sys import exit, argv
from random import choice, shuffle
from itertools import product
from matplotlib.pyplot import plot, ylabel, show, legend, xlabel, ylim

#PAREMETERS:
###################################
epoch_num = 1000 #Number of epochs
reps = 15
dropout_prob = 0.75
###################################

##BUILD TRAINING DATA##
FEAT_CONVERT = {	     #syll	son		voice	cor		cont	lab		vel 	nasal 	high	low		tense	ejective
					"p": [-1.0,	-1.0,	-1.0,	-1.0, 	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"b": [-1.0,	-1.0,	1.0,	-1.0, 	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"t": [-1.0,	-1.0,	-1.0,	1.0, 	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"d": [-1.0,	-1.0,	1.0,	1.0, 	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"c": [-1.0,	-1.0,	-1.0,	1.0, 	-1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1],
					"%": [-1.0,	-1.0,	1.0,	1.0, 	-1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1],
					"k": [-1.0,	-1.0,	-1.0,	-1.0, 	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"g": [-1.0,	-1.0,	1.0,	-1.0, 	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"f": [-1.0,	-1.0,	-1.0,	-1.0, 	1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"v": [-1.0,	-1.0,	1.0,	-1.0, 	1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"s": [-1.0,	-1.0,	-1.0,	1.0, 	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"z": [-1.0,	-1.0,	1.0,	1.0, 	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"!": [-1.0,	-1.0,	-1.0,	1.0, 	1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1],
					"&": [-1.0,	-1.0,	1.0,	1.0, 	1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1],
					"x": [-1.0,	-1.0,	-1.0,	-1.0, 	1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"y": [-1.0,	-1.0,	1.0,	-1.0, 	1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"P": [-1.0,	-1.0,	-1.0,	-1.0, 	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"B": [-1.0,	-1.0,	1.0,	-1.0, 	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"T": [-1.0,	-1.0,	-1.0,	1.0, 	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"D": [-1.0,	-1.0,	1.0,	1.0, 	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"C": [-1.0,	-1.0,	-1.0,	1.0, 	-1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	1],
					"^": [-1.0,	-1.0,	1.0,	1.0, 	-1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	1],
					"K": [-1.0,	-1.0,	-1.0,	-1.0, 	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"G": [-1.0,	-1.0,	1.0,	-1.0, 	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"F": [-1.0,	-1.0,	-1.0,	-1.0, 	1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"V": [-1.0,	-1.0,	1.0,	-1.0, 	1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"S": [-1.0,	-1.0,	-1.0,	1.0, 	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"Z": [-1.0,	-1.0,	1.0,	1.0, 	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"@": [-1.0,	-1.0,	-1.0,	1.0, 	1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	1],
					"*": [-1.0,	-1.0,	1.0,	1.0, 	1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	1],
					"X": [-1.0,	-1.0,	-1.0,	-1.0, 	1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"Y": [-1.0,	-1.0,	1.0,	-1.0, 	1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1],
					"w": [-1.0,	1.0,	1.0,	-1.0, 	1.0,	1.0,	1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1],
					"j": [-1.0,	1.0,	1.0,	1.0, 	1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1],
					"l": [-1.0,	1.0,	1.0,	1.0, 	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"i": [1.0,	1.0,	1.0,	-1.0, 	1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	1.0,	-1],
					"o": [1.0,	1.0,	1.0,	-1.0, 	1.0,	1.0,	1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1],
					"e": [1.0,	1.0,	1.0,	-1.0, 	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1],
					"u": [1.0,	1.0,	1.0,	-1.0, 	1.0,	1.0,	1.0,	-1.0,	1.0,	-1.0,	1.0,	-1],
					"I": [1.0,	1.0,	1.0,	-1.0, 	1.0,	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1],
					"O": [1.0,	1.0,	1.0,	-1.0, 	1.0,	1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"E": [1.0,	1.0,	1.0,	-1.0, 	1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1.0,	-1],
					"U": [1.0,	1.0,	1.0,	-1.0, 	1.0,	1.0,	1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1],
				}
withheld_segs = [	
						 [-1.0,	1.0,	1.0,	1.0, 	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1],#[n]
						 [1.0,	1.0,	1.0,	-1.0, 	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1] #[a]
				]
inventory = list(FEAT_CONVERT.values())
feat_num = len(inventory[0])

	
#Construct syllables:
vowels = [seg for seg in inventory if seg[0] == 1.0]
consonants = [seg for seg in inventory if seg[0] == -1.0]
syllables = list(product(consonants, vowels)) #Find all possible syllables using this inventory
syllables = [list(syll) for syll in syllables] #Convert syllables to lists

##RUN SIMULATIONS##	
train_accuracies = []
test_accuracies = []
loss_to_save = []
correctness_to_save = []
learning_curves = []
for rep in range(reps):
	print (" Rep: ", str(rep))
	
	#Build the train/test data:
	shuffle(syllables) #Shuffle the syllables
	withheld_syll = [withheld_segs[0], withheld_segs[1]]
	X = np.array(syllables)
	Y = np.array([syll+syll for syll in syllables])
	
	#Build the model:
	model = Seq2Seq(input_dim=feat_num, hidden_dim=feat_num*3,
					output_length=len(Y[0]), output_dim=feat_num,
					depth=2, dropout=dropout_prob)
					
	model.compile(	loss='mse', 
					optimizer='rmsprop'
				  )
	
	#Train the model:
	hist = model.fit(
						 X, Y,
						 epochs=epoch_num,
						 batch_size=len(X)
					 )
	learning_curves.append(hist.history["loss"])
					 
	#Test the model on trained data:
	trained_IN = np.tile(X[0], (1, 1, 1))
	trained_OUT = np.tile(Y[0], (1, 1, 1))
	train_pred = model.predict(trained_IN)
	
	#Test the model on withheld data:
	withheld_IN = np.tile(np.array(withheld_syll), (1, 1, 1))
	withheld_OUT = np.tile(np.array(withheld_syll+withheld_syll), (1, 1, 1))
	withheld_pred = model.predict(withheld_IN)
		
	#Save whether the model is doing reduplication correctly, according to our decision rule:
	test_Correct = []
	train_Correct = []
	for i, seg in enumerate(train_pred[0]):
		for j, feat in enumerate(seg):
			testPred_isPlus = withheld_pred[0][i][j] > 0
			trainPred_isPlus = train_pred[0][i][j] > 0
			
			trainReal_isPlus = trained_OUT[0][i][j] > 0
			testReal_isPlus = withheld_OUT[0][i][j] > 0
			
			test_Correct.append(int(testPred_isPlus == testReal_isPlus))
			train_Correct.append(int(trainPred_isPlus == trainReal_isPlus))
	correctness_to_save.append([int(0 not in train_Correct), int(0 not in test_Correct)])		


#Print a sample datum:
print ("Test model for largest inventory...")
print ("Trained input: ", trained_IN)
print ("Predicted output: ", train_pred)
print ("Withheld input: ", withheld_IN)
print ("Predicted output: ", withheld_pred)

#Save all the reps for analysis in R:
cts = np.array(correctness_to_save)
np.savetxt("Novel Features_CORRECTNESS_Dropout="+str(dropout_prob)+".csv", cts, delimiter=",", newline="\n")

#Plot an average learning curve:
plot(np.mean(learning_curves, axis=0))
show()