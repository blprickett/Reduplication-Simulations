# Reduplication Experiment Simulations

The script "marcus_simulations.py" is made to simulate the experiments from Marcus et al. (1999) with a Sequence-to-Sequence neural network (Sutskever et al. 2014). The script doesn't take any input files, but it does create output files in the directory in which it's run. For more on this software and the theoretical questions that motivated it, see §4 of [this manuscript](https://works.bepress.com/joe_pater/38/).

The script "generalization_scope_tests.py" is made to more carefully test how the model generalizes after learning a reduplicative pattern. To do this, Berent's (2013) *scopes of generalization* are used: novel syllables (or words), novel segments, and novel feature values. The script doesn't take any input files, but it does create output files in the directory in which it's run. For more on this software and the theoretical questions that motivated it, see §5 of [this manuscript](https://works.bepress.com/joe_pater/38/).

## Dependencies

To run these scripts, you'll need Python 3 and the following Python packages:

* numpy
* scipy
* tensorflow
* keras
* sys
* random
* matplotlib
* itertools

(Note that an earlier version of these scripts used recurrentshop from [this repo](https://github.com/blprickett/recurrentshop) and seq2seq from [this repo](https://github.com/farizrahman4u/seq2seq). However, these have been replaced with the Seq2Seq.py file included in this repo. If you get slightly different results from what we report in the paper, this difference in likely the reason).

## Command line arguments

The script "marcus_simulations.py" takes the following arguments (in this order):

* Experiment: this corresponds to which experiment you want to simulate (should be "1", "2", or "3"). Techinally, "3" simulates the replication of Marcus et al.'s (1999) experiment that was performed by Endress et al. (2007).
* Repetition number: this corresponds to how many repetitions of the simulation you want to run.
* Epoch number: this corresponds to how many epochs you want in pretraining (per repetition). This number is halved to produce the number of epochs in training for the experiment simulation.
* Dropout probability: how likely is each unit in the model to be dropped out during training?
* Pattern: which pattern is present in training? For Experiments 1 and 2, this should be either "ABB" or "ABA". For Experiment 3, it needs to be either "ABB" or "AAB". 
* Vocabulary size: how many randomly-created words are present in pretraining?
* Reduplication probability in pretraining: what proportion (above chance) of the pretraining contains "ABB" reduplicated forms?

So, for example, if you wanted to simulate Marcus et al.'s second experiment's ABB condition, for five repetitions, each of which had 1000 epochs of pretraining (and 500 epochs of training), with a model whose units drop out 50% of the time, with a pretraining vocabulary of 1000 words and no reduplication added to that pretraining, you would run the following command line:

```bash
python marcus_simulations.py 2 5 1000 .5 ABB 1000 0
```

The script "generalization_scope_tests.py" takes the following arguments (in this order):

* Epoch number: this corresponds to how many epochs you want in pretraining (per repetition). This number is halved to produce the number of epochs in training for the experiment simulation.
* Repetition number: this corresponds to how many repetitions of the simulation you want to run.
* Dropout probability: how likely is each unit in the model to be dropped out during training?
* Scope: which scope of generalization are we testing the model on? Should be either "feature" for novel feature values, "segment" for novel segments, or "syllable" for novel syllables.

So, for example, if you wanted to run a five-repetition-long test on the model's ability to generalize to novel segments after training for 100 epochs with a dropout probability of .5, you would run the following command line:

```bash
python generalization_scope_tests.py 100 5 .5 segment
```

## References
* Berent, I. (2013). The phonological mind. *Trends in cognitive sciences, 17(7)*, 319-327.
* Endress, A. D., Dehaene-Lambertz, G., & Mehler, J. (2007). Perceptual constraints and the learnability of simple grammars. *Cognition, 105(3)*, 577-614.
* Marcus, G. F., Vijayan, S., Rao, S. B., & Vishton, P. M. (1999). Rule learning by seven-month-old infants. *Science, 283(5398)*, 77-80.
* Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In *Advances in neural information processing systems* (pp. 3104-3112).
