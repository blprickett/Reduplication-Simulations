import numpy as np
import keras
import Seq2Seq
from sys import argv
from random import choice, shuffle
from itertools import product
from matplotlib.pyplot import plot, ylabel, show, legend, xlabel, title

#PAREMETERS:
###################################
EPOCHS = int(argv[1]) #Number of epochs
REPS = int(argv[2]) #Number of repetitions to run
DROPOUT = float(argv[3]) #Probability of dropout in the model
SCOPE = argv[4] #Must be from the set {"feature", "segment", "syllable"}
###################################

def novel_feat_data ():
	#Create these features by hand for better control:
	feat_convert = {	  #syll	son		voice	cor		cont	lab		vel 	nasal 	high	low		tense	ejective
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
	
	#Withheld segment is always [n] (since it's value for nasal is 1.0)
	withheld_seg = [-1.0,	1.0,	1.0,	1.0, 	-1.0,	-1.0,	-1.0,	1.0,	-1.0,	-1.0,	-1.0,	-1] #[n]
	
	#Get the list of segments and the number of features:
	inventory = list(feat_convert.values())
	feat_num = len(inventory[0])
	
	#Create the syllables that are in our training data:
	vowels = [seg for seg in inventory if seg[0] == 1.0]
	consonants = [seg for seg in inventory if seg[0] == -1.0]
	syllables = list(product(consonants, vowels)) #Find all possible syllables using this inventory
	shuffle(syllables)
	syllables = [list(syll) for syll in syllables] #Convert syllables to lists

	#Create the withheld syllable:
	withheld_syll = [withheld_seg, vowels[0]]	
	
	return feat_num, withheld_syll, syllables
	
def novel_seg_data (seg_num=40):
	#Calculate number of contrastive features that are necessary:
	feat_num = int(np.ceil(np.log2(seg_num))) 
	
	#Create all possible feature bundles:
	inventory = list(product([1.0, -1.0], repeat=feat_num)) 
	
	#Shuffle all the feature bundles so those that we're left with are random:
	shuffle(inventory) 
	
	#Cut out all the extra feature bundles and convert them to a list
	inventory = [list(inventory[seg]) for seg in range(len(inventory)) if seg < seg_num] 

	#This ensures that every inventory has at least one vowel:
	vowel_check = False
	for segment in inventory:
		if segment[0] == 1.0: #The first feature will always be [syllabic]
			vowel_check = True
			break
	if not vowel_check:
		inventory[0][0] = 1.0
		
	#Create the syllables that are in our training data:
	vowels = [seg for seg in inventory if seg[0] == 1.0]
	consonants = [seg for seg in inventory if seg[0] == -1.0]
	withheld_seg = consonants.pop()
	syllables = list(product(consonants, vowels)) #Find all possible syllables using this inventory
	shuffle(syllables) #Shuffle the syllables
	syllables = [list(syll) for syll in syllables] #Convert syllables to lists

	#Create the withheld syllable:
	withheld_syll = [withheld_seg, vowels[0]]
	
	return feat_num, withheld_syll, syllables
	
def novel_syll_data (seg_num=40):
	#Calculate number of contrastive features that are necessary:
	feat_num = int(np.ceil(np.log2(seg_num))) 
	
	#Create all possible feature bundles:
	inventory = list(product([1.0, -1.0], repeat=feat_num)) 
	
	#Shuffle all the feature bundles so those that we're left with are random:
	shuffle(inventory) 
	
	#Cut out all the extra feature bundles and convert them to a list
	inventory = [list(inventory[seg]) for seg in range(len(inventory)) if seg < seg_num] 

	#This ensures that every inventory has at least one vowel:
	vowel_check = False
	for segment in inventory:
		if segment[0] == 1.0: #The first feature will always be [syllabic]
			vowel_check = True
			break
	if not vowel_check:
		inventory[0][0] = 1.0
		
	#Create the syllables that are in our training data:
	vowels = [seg for seg in inventory if seg[0] == 1.0]
	consonants = [seg for seg in inventory if seg[0] == -1.0]
	syllables = list(product(consonants, vowels)) #Find all possible syllables using this inventory
	shuffle(syllables) #Shuffle the syllables
	syllables = [list(syll) for syll in syllables] #Convert syllables to lists

	#Create the withheld syllable:
	withheld_syll = syllables.pop()
	
	return feat_num, withheld_syll, syllables
	
##RUN SIMULATIONS##	
correctness_to_save = []
learning_curves = []
for rep in range(REPS):
	print (" Rep: ", str(rep))
	
	#Erase the previous model:
	keras.backend.clear_session()
	
	#Build the train/test data:
	if SCOPE == "feature":
		feat_num, withheld_syll, syllables = novel_feat_data()
	elif SCOPE == "segment":
		feat_num, withheld_syll, syllables = novel_seg_data()
	elif SCOPE == "syllable":
		feat_num, withheld_syll, syllables = novel_syll_data()
	else:
		raise Exception("Wrong scope! Must be from the set {feature, segment, syllable}.")
	
	
	X = np.array(syllables)
	Y = np.array([syll+syll for syll in syllables])
	
	#Build the model:
	model = Seq2Seq.seq2seq(
						input_dim=feat_num,
						hidden_dim=feat_num*3,
						output_length=Y.shape[1],
						output_dim=Y.shape[2],
            					batch_size=1,
            					learn_rate=0.001,
					  	layer_type="lstm",
            					dropout=DROPOUT
					)
	
	#Train the model:
	hist = model.train(
						 X, 
						 Y,
						 epoch_num=EPOCHS,
						 print_every=10
					 )
	learning_curves.append(hist["Loss"])
					 
	#Test the model on trained data:
	trained_IN = np.tile(X[0], (1, 1, 1))
	trained_OUT = np.tile(Y[0], (1, 1, 1))
	train_pred = model.predict(trained_IN)
	
	#Test the model on withheld data:
	withheld_IN = np.tile(np.array(withheld_syll), (1, 1, 1))
	withheld_OUT = np.tile(np.array(withheld_syll+withheld_syll), (1, 1, 1))
	withheld_pred = model.predict(withheld_IN)
		
	#Go through each feature in each segment of the output and check to see
	#if it's correct:
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
	#Save whether or not the model was perfectly correct in training and testing:
	#(1=success and 0=failure, so you can average the columns for overall proportion of success)
	correctness_to_save.append([int(0 not in train_Correct), int(0 not in test_Correct)])		


#Save all the reps for analysis:
cts = np.array(correctness_to_save)
np.savetxt(
				"novel_"+SCOPE+"_Dropout="+str(DROPOUT)+".csv",
				cts,
				delimiter=",",
				header="Training Data Successes,Withheld Data Successes",
				newline="\n",
				comments=""
		   )

#Plot an average learning curve:
plot(np.mean(learning_curves, axis=0))
xlabel("Epoch")
ylabel("Loss")
title("Average Loss Across Repetitions")
show()
