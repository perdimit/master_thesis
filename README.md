# master_thesis
Per-Dimitri's master thesis.

**Models**
*Training*
The Models folder includes main_training.py.
Here one can run a training run or a hyperparameter tuning run.
- Load the data set, drop features not needed.
- Initialize prior and pmf. e.g. use the lines already there. Initialize model.
- Set hyperparameter lists e.g. epocs = [1000, 2000], learning rate = [0.01, 0.0001]. Make a dictionary of the parameters
params = {'epochs': epochs, 'learning_rate': learning_rate}
- Run model.multiple_runs(params). This will run over all permutations of the hyperparameters.

*Testing*
The Models folder includes main_testing.py.
Here one can load parameters from a training run and run a test run. Where more data is dedicated to training. Test data must be passed in.

**Visualization**
This folder includes various plotting scripts.

**Data Harvest**
Creates a file in pandas-style with feature and target columns from a folder of SLHA files.
