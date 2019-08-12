# Reduplication Experiment Simulations

The script "marcus_simulations.py" is made to simulate the experiments from Marcus et al. (1999) with a Sequence-to-Sequence neural network (Sutskever et al. 2014). The script doesn't take any input files, but it does create output files in the directory in which it's run. For more on this software and the theoretical questions that motivated it, see §4 of [this paper](https://works.bepress.com/joe_pater/38/).

## Dependencies

To run "marcus_simulations.py", you'll need the following Python packages:

* numpy
* scipy
* keras
* seq2seq (from [this repo](https://github.com/farizrahman4u/seq2seq))
* sys
* random
* matplotlib

## Command line arguments

The script "marcus_simulations.py" takes the following arguments (in this order):

* Experiment: this corresponds to which experiment you want to simulate should be "1", "2", or "3". Techinally, "3" simulates the replication of Marcus et al.'s (1999) experiment that was performed by Endress et al. (2007).
* Repetition number: this corresponds to how many repetitions of the simulation you want to run.
* Epoch number: this corresponds to how many epochs you want in pretraining (per repetition). This number is halved to produce the number of epochs in training for the experiment simulation.
* Dropout probability: how likely is each unit in the model to be dropped out during training?
* Pattern: which pattern is present in training? For Experiments 1 and 2, this should be either "ABB" or "ABA". For Experiment 3, it needs to be either "ABB" or "AAB". 
* Vocabulary size: how many randomly-created words are present in pretraining?
* Reduplication probability in pretraining: what proportion (above chance) of the pretraining contains "ABB" reduplicated forms?

## References
* Endress, A. D., Dehaene-Lambertz, G., & Mehler, J. (2007). Perceptual constraints and the learnability of simple grammars. *Cognition, 105(3)*, 577-614.
* Marcus, G. F., Vijayan, S., Rao, S. B., & Vishton, P. M. (1999). Rule learning by seven-month-old infants. *Science, 283(5398)*, 77-80.
* Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In *Advances in neural information processing systems* (pp. 3104-3112).
