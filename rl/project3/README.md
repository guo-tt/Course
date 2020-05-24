## Project 3: RoboCup

The repo contains code to run experiments required for CS 7642 Summer 2018 Project 3.

* Required packages: Python 3.6.5, Numpy 1.15.5, Pandas 0.23.1, Matplotlib 2.2.2, cvxopt 1.2.0, progress 1.4.
	* progress 1.4  can be found [here](https://pypi.org/project/progress), and source code [here](https://stackoverflow.com/questions/3160699/python-progress-bar).
* The repo contains these folders:
	* `results`: contains csv files that are results from each experiments, Q tables in the format of `.npy`, Q-value difference plots, as well as `log.csv` recording parameters used in all experiments. **This folder needs to be created before any experiment can be run**.
	* `figs`: plots used in figures in the report.
	* `report`: full report in PDF and LaTex format, and accessory files

* The repo contains these files:
	* `README.md`: this file, description of this repo
	* `Q.py`: Q learner, with different variants, called from `run.py`. All learner classes includes both `learn()` and `train()` methods.
	* `friendQ.py`: CE-Q learner, with different variants.
	* `foeQ.py`: foe-Q learner, with different variants.
	* `ceQ.py`: CE-Q learner, with different variants.
	* `run.py`: takes parameters as argument and run experiments
	* `run.sh`: bash script to run experiments with different parameters in batch.
	* `util.py`: utility functions, including plotting and logging functions, and linear programming functions for maximin and CE.
	* `soccer.py`: soccer game environment.
	
	
* Perform experiments by running `python3 run.py <algo_name> <init_lr> <end_lr> <init_eps> <end_eps> <max_iter>` from terminal, or running `bash run.sh` to perform experiments in batch.