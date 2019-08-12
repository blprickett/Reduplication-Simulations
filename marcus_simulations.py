import numpy as np
import scipy.stats as sp
import keras
import seq2seq
from seq2seq.models import Seq2Seq
from sys import argv
from random import choice, shuffle
from matplotlib.pyplot import plot, ylabel, show, legend, xlabel

####PARAMETERS:#################
EXP = argv[1] #Choose between '1', '2', and '3' 
REPS = int(argv[2]) #Number of repititions to run
EPOCHS = int(argv[3]) #Number of epochs to use in training
DROPOUT = float(argv[4]) #Dropout probability
PATTERN = argv[5] #Choose between "ABB" and "ABA"
VOCAB_SIZE = int(argv[6]) #Number of words in pretraining 
REDUP_IN_PT = float(argv[7]) #What's the probability of seeing reduplication in pretraining (this pretends that chance is 0)
################################ 

#Create the correct data for the given experiment:
if EXP == '1':
	train_ABA = [
					"gatiga", "ganaga", "gagiga", "galaga", 
					"litili", "ligili", "lilali", "nigini", 
					"ninani", "nilani", "talata", "tatita", 
					"linali", "nitini", "tanata", "tagita"
				] * 3
	train_ABB = [
					"tigaga", "nagaga", "gigaga", "lagaga", 
					"tilili", "gilili", "lalili", "ginini", 
					"nanini", "lanini", "latata", "titata", 
					"nalili", "tinini", "natata", "gitata"
				] * 3
	test_ABB = ["wofefe", "dekoko"] * 3
	test_ABA = ["wofewo", "dekode"] * 3
	
	FEAT_CONVERT = {	#syll	son		voice	cor		cont	lab		vel 	nasal 	high	low 	back
						"t": [-1.0,	-1.0,	-1.0,	1.0, 	-1.0,	0.0,	0.0,	-1.0,	0.0,	0.0, 	0.0],
						"d": [-1.0,	-1.0,	1.0,	1.0, 	-1.0,	0.0,	0.0,	-1.0,	0.0,	0.0, 	0.0],
						"k": [-1.0,	-1.0,	-1.0,	0.0, 	-1.0,	0.0,	1.0,	-1.0,	0.0,	0.0, 	0.0],
						"g": [-1.0,	-1.0,	1.0,	0.0, 	-1.0,	0.0,	1.0,	-1.0,	0.0,	0.0, 	0.0],
						"f": [-1.0,	-1.0,	-1.0,	0.0, 	1.0,	1.0,	0.0,	-1.0,	0.0,	0.0, 	0.0],
						"w": [-1.0,	1.0,	0.0,	0.0, 	0.0,	1.0,	1.0,	-1.0,	1.0,	-1.0, 	1.0],
						"l": [-1.0,	1.0,	0.0,	1.0, 	0.0,	0.0,	0.0,	-1.0,	0.0,	0.0, 	0.0],
						"n": [-1.0,	1.0,	0.0,	1.0, 	0.0,	0.0,	0.0,	1.0,	0.0,	0.0, 	0.0],
						"a": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	-1.0,	-1.0,	1.0, 	1.0],
						"i": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	-1.0,	1.0,	-1.0, 	-1.0],
						"o": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	-1.0,	-1.0,	-1.0, 	1.0],
						"e": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	-1.0,	-1.0,	-1.0, 	-1.0]

					}
elif EXP == '2':
	train_ABA = [
					"ledile", "lejele", "lelile", "lewele", 
					"widiwi", "wijewi", "wiliwi", "wiwewi", 
					"jidiji", "jijeji", "jiliji", "jiweji", 
					"dedide", "dejede", "delide", "dewede"
				] * 3
	train_ABB = [
					"dilele", "jelele", "lilele", "welele", 
					"diwiwi", "jewiwi", "liwiwi", "wewiwi", 
					"dijiji", "jejiji", "lijiji", "wejiji", 
					"didede", "jedede", "lidede", "wedede"
				] * 3
	test_ABB = ["bapopo", "kogaga"] * 3
	test_ABA = ["bapoba", "kogako"] * 3
	
	FEAT_CONVERT = {	#syll	son		voice	cor		cont	lab		vel 	ant 	high	low 	back
					"p": [-1.0,	-1.0,	-1.0,	0.0, 	-1.0,	1.0,	0.0,	0.0,	0.0,	0.0, 	0.0],
					"b": [-1.0,	-1.0,	1.0,	0.0, 	-1.0,	1.0,	0.0,	0.0,	0.0,	0.0, 	0.0],
					"t": [-1.0,	-1.0,	-1.0,	1.0, 	-1.0,	0.0,	0.0,	1.0,	0.0,	0.0, 	0.0],
					"d": [-1.0,	-1.0,	1.0,	1.0, 	-1.0,	0.0,	0.0,	1.0,	0.0,	0.0, 	0.0],
					"k": [-1.0,	-1.0,	-1.0,	0.0, 	-1.0,	0.0,	1.0,	0.0,	0.0,	0.0, 	0.0],
					"g": [-1.0,	-1.0,	1.0,	0.0, 	-1.0,	0.0,	1.0,	0.0,	0.0,	0.0, 	0.0],
					"f": [-1.0,	-1.0,	-1.0,	0.0, 	1.0,	1.0,	0.0,	0.0,	0.0,	0.0, 	0.0],
					"j": [-1.0,	-1.0,	1.0,	1.0, 	-1.0,	0.0,	0.0,	-1.0,	1.0,	0.0, 	0.0],
					"w": [-1.0,	1.0,	0.0,	0.0, 	0.0,	1.0,	1.0,	0.0,	1.0,	-1.0, 	1.0],
					"l": [-1.0,	1.0,	0.0,	1.0, 	0.0,	0.0,	0.0,	1.0,	0.0,	0.0, 	0.0],
					"a": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	0.0,	-1.0,	1.0, 	1.0],
					"i": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	0.0,	1.0,	-1.0, 	-1.0],
					"o": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	0.0,	-1.0,	-1.0, 	1.0],
					"e": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	0.0,	-1.0,	-1.0, 	-1.0]
				}
elif EXP == '3':
	train_AAB = [
					"leledi", "leleje", "leleli", "lelewe", 
					"wiwidi", "wiwije", "wiwili", "wiwiwe", 
					"jijidi", "jijije", "jijili", "jijiwe", 
					"dededi", "dedeje", "dedeli", "dedewe"
				] * 3
	train_ABB = [
					"dilele", "jelele", "lilele", "welele", 
					"diwiwi", "jewiwi", "liwiwi", "wewiwi", 
					"dijiji", "jejiji", "lijiji", "wejiji", 
					"didede", "jedede", "lidede", "wedede"
				] * 3
	test_ABB = ["bapopo", "kogaga"] * 3
	test_AAB = ["babapo", "kokoga"] * 3
	test_AAA = ["bababa", "kokoko"] * 3
	
	FEAT_CONVERT = {		  #syll	son		voice	cor		cont	lab		vel 	ant 	high	low 	back
						"p": [-1.0,	-1.0,	-1.0,	0.0, 	-1.0,	1.0,	0.0,	0.0,	0.0,	0.0, 	0.0],
						"b": [-1.0,	-1.0,	1.0,	0.0, 	-1.0,	1.0,	0.0,	0.0,	0.0,	0.0, 	0.0],
						"t": [-1.0,	-1.0,	-1.0,	1.0, 	-1.0,	0.0,	0.0,	1.0,	0.0,	0.0, 	0.0],
						"d": [-1.0,	-1.0,	1.0,	1.0, 	-1.0,	0.0,	0.0,	1.0,	0.0,	0.0, 	0.0],
						"k": [-1.0,	-1.0,	-1.0,	0.0, 	-1.0,	0.0,	1.0,	0.0,	0.0,	0.0, 	0.0],
						"g": [-1.0,	-1.0,	1.0,	0.0, 	-1.0,	0.0,	1.0,	0.0,	0.0,	0.0, 	0.0],
						"f": [-1.0,	-1.0,	-1.0,	0.0, 	1.0,	1.0,	0.0,	0.0,	0.0,	0.0, 	0.0],
						"j": [-1.0,	-1.0,	1.0,	1.0, 	-1.0,	0.0,	0.0,	-1.0,	0.0,	0.0, 	0.0],
						"w": [-1.0,	1.0,	0.0,	0.0, 	0.0,	1.0,	1.0,	0.0,	1.0,	-1.0, 	1.0],
						"l": [-1.0,	1.0,	0.0,	1.0, 	0.0,	0.0,	0.0,	1.0,	0.0,	0.0, 	0.0],
						"a": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	0.0,	-1.0,	1.0, 	1.0],
						"i": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	0.0,	1.0,	-1.0, 	-1.0],
						"o": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	0.0,	-1.0,	-1.0, 	1.0],
						"e": [1.0,	1.0,	0.0,	0.0, 	0.0,	0.0,	0.0,	0.0,	-1.0,	-1.0, 	-1.0],
						"_": [0.0,  0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0]
					}	
else:
	raise Exception("Wrong experiment number used. Must come from the set: {1, 2, 3}.")
	
FEAT_NUM = len(list(FEAT_CONVERT.values())[0])
	
#Break down the experimental data into chunks, to randomly produce pretraining
if EXP == "3":
	full_set = train_AAB+test_AAB+train_ABB+test_ABB
else:
	full_set = train_ABA+test_ABA+train_ABB+test_ABB
sylls = []
for word in full_set:
	syll1 = word[0]+word[1]
	syll2 = word[2]+word[3]
	syll3 = word[4]+word[5]
	sylls += [syll1, syll2, syll3]
all_sylls = list(set(sylls))

#Converting the experimental data to mappings of vectors (how we do this depends on the experiment)
if PATTERN == "ABB":
	this_td = train_ABB
elif PATTERN == "ABA":
	this_td = train+ABA
elif PATTERN == "AAB":
	this_td = train_AAB
else:
	raise Exception("Wrong pattern used. Must come from the set: {ABB, ABA, AAB}.")
	
if EXP == "3":
	empty_syll = "__"
	vec_empty_syll = [FEAT_CONVERT[seg] for seg in empty_syll]

	raw_X = []
	raw_Y = []
	indeces = []
	for index, datum in enumerate(this_td):
		vectorized_datum = [FEAT_CONVERT[seg] for seg in datum]
		x = vectorized_datum[:2]+vec_empty_syll+vectorized_datum[-2:]
		y = vectorized_datum[2:-2]
		raw_X.append(x)
		raw_Y.append(y)
		indeces.append(index)

	print("Processing testing data...")
	test_ABB_x = [[FEAT_CONVERT[seg] for seg in datum[:2]+empty_syll+datum[-2:]] for datum in test_ABB]
	test_ABB_y = [[FEAT_CONVERT[seg] for seg in datum[2:-2]] for datum in test_ABB]

	test_AAB_x = [[FEAT_CONVERT[seg] for seg in datum[:2]+empty_syll+datum[-2:]] for datum in test_AAB]
	test_AAB_y = [[FEAT_CONVERT[seg] for seg in datum[2:-2]] for datum in test_AAB]

	test_AAA_x = [[FEAT_CONVERT[seg] for seg in datum[:2]+empty_syll+datum[-2:]] for datum in test_AAA]
	test_AAA_y = [[FEAT_CONVERT[seg] for seg in datum[2:-2]] for datum in test_AAA]
	
	aaa_losses = []
	aab_losses = []
else:
	raw_X = []
	raw_Y = []
	indeces = []
	for index, datum in enumerate(this_td):
		vectorized_datum = [FEAT_CONVERT[seg] for seg in datum]
		x = vectorized_datum[:4]
		y = vectorized_datum[4:]
		raw_X.append(x)
		raw_Y.append(y)
		indeces.append(index)

	test_ABB_x = np.array([[FEAT_CONVERT[seg] for seg in datum[:4]] for datum in test_ABB])
	test_ABB_y = np.array([[FEAT_CONVERT[seg] for seg in datum[4:]] for datum in test_ABB])
	test_ABA_x = np.array([[FEAT_CONVERT[seg] for seg in datum[:4]] for datum in test_ABA])
	test_ABA_y = np.array([[FEAT_CONVERT[seg] for seg in datum[4:]] for datum in test_ABA])
	test_indices = list(range(len(test_ABB_x)))
	
	aba_losses = []

#Running the simulations:
learning_curves = []
pretraining_curves = []
abb_losses = []
training_losses = []
for rep in range(REPS):
	print ("Rep: "+str(rep))
	
	#Erase the previous model:
	keras.backend.clear_session() 
  
	#Build the new model:
	model = Seq2Seq(
						input_dim=FEAT_NUM,
						hidden_dim=FEAT_NUM,
						output_length=2,
						output_dim=FEAT_NUM,
						depth=2,
						dropout=DROPOUT
					)
					
	model.compile(	
					loss='mse', 
					optimizer='rmsprop'
				  )
				  
	#PRETRAINING
	if VOCAB_SIZE > 0:
		print ("Simulating real-life experience of infants...Rep="+str(rep))
		irl_X = []
		irl_Y = []
		for word in range(VOCAB_SIZE):
		##Putting reduplication in training:
			if np.random.rand() < REDUP_IN_PT:
				syll_alpha = choice(all_sylls)
				#template = choice(["ABB", "ABA"])
				#if template == "ABB":
				#	string_X = choice(all_sylls)+syll_alpha
				#elif template == "ABA":
				#	string_X = syll_alpha+choice(all_sylls)
		  
				#string_X = syll_alpha+syll_alpha
		  
				#string_X = syll_alpha+choice(all_sylls)  #ABA
				string_X = choice(all_sylls)+syll_alpha #ABB
		  
				#template = choice(["wofewo", "dekode"])
				#string_X = template[:4]
				#string_Y = template[4:]
				
				string_Y = syll_alpha
			else:
			  string_X = choice(all_sylls)+choice(all_sylls)
			  string_Y = choice(all_sylls)

			vector_X = [FEAT_CONVERT[seg] for seg in string_X]
			vector_Y = [FEAT_CONVERT[seg] for seg in string_Y]
			
			irl_X.append(vector_X)
			irl_Y.append(vector_Y)
		irl_X = np.array(irl_X)
		irl_Y = np.array(irl_Y)
		decoder_init = model.decoder.get_weights()
		irl_curve = model.fit(
								irl_X, 
								irl_Y,
								epochs=EPOCHS,
								batch_size=50,
								verbose=0
							  )
		model.decoder.set_weights(decoder_init) #Set the decoder back to its initial weights.
		pretraining_curves.append(irl_curve.history["loss"])
	
	
	#SIMULATE EXPERIMENT
	#Randomize training order:
	shuffle(indeces)
	shuffled_X = [raw_X[i] for i in indeces]
	shuffled_Y = [raw_Y[i] for i in indeces]
	
	#Convert to numpy arrays:
	X = np.array(shuffled_X)
	Y = np.array(shuffled_Y)
	
	#Train the model:
	print("Simulating experiment...Rep="+str(rep))
	hist = model.fit(
						X, 
						Y,
						epochs=int(np.ceil(EPOCHS/2)),
						batch_size=50,
						verbose=0
					 )
	learning_curves.append(hist.history["loss"])
	
	#Marcus et al's testing phase:
	print("Testing model...Rep="+str(rep))
	
	abb_loss = model.test_on_batch(test_ABB_x, test_ABB_y)
	abb_losses.append(str(abb_loss))
	
	training_loss = model.test_on_batch(X, Y)
	training_losses.append(str(training_loss))
	
	if EXP == "3":
		aab_loss = model.test_on_batch(test_AAB_x, test_AAB_y)
		aab_losses.append(str(aab_loss))
	
		aaa_loss = model.test_on_batch(test_AAA_x, test_AAA_y)
		aaa_losses.append(str(aaa_loss))
	else:
		aba_loss = model.test_on_batch(test_ABA_x, test_ABA_y)
		aba_losses.append(str(aba_loss))
	
#Print correctness over all reps:
if EXP == "3":
	OUT_file = open("Marcus Simulations (Exp "+EXP+" - Dropout="+str(DROPOUT)+", Pattern="+PATTERN+", "+str(REDUP_IN_PT)+").csv", "w")
	OUT_file.write("Loss for ABB Test Data,Loss for AAB Test Data,Loss for AAA Test Data,Loss on Training Data")
	OUT_file.write("\n")
	for trial in range(len(abb_losses)):
		OUT_file.write(",".join([	
									abb_losses[trial],
									aab_losses[trial], 
									aaa_losses[trial],
									training_losses[trial]
								]))
		OUT_file.write("\n")
else:
	OUT_file = open("Marcus Simulations (Exp "+EXP+" - Dropout="+str(DROPOUT)+", Pattern="+PATTERN+", Pretraining Redup Prob="+str(REDUP_IN_PT)+").csv", "w")
	OUT_file.write("Loss for ABB Test Data,Loss for ABA Test Data,Loss on Training Data")
	OUT_file.write("\n")
	for trial in range(len(abb_losses)):
		OUT_file.write(",".join([	
									abb_losses[trial],
									aba_losses[trial], 
									training_losses[trial]
								]))
		OUT_file.write("\n")
OUT_file.close()
 
#Plot out the average learning curve:	
average_curves = np.mean(learning_curves, axis=0)
average_pt_curves = np.mean(pretraining_curves, axis=0)
plot(average_curves, label="Experiment Simulation")
plot(average_pt_curves, label="Pretraining")
legend()
ylabel("Loss")
xlabel("Epoch")
show()