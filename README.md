# InContextTD

Welcome to the InContextTD repository, which accompanies the paper: [Transformers Learn Temporal Difference Methods for In-Context Reinforcement Learning](https://arxiv.org/abs/2405.13861).

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Custom Experiments](#custom-experiment-settings)
  - [Demo](#demo)
  - [Theory Verification](#theory-verification)
  - [Complete Replication](#complete-replication)
  - [Test](#test)
- [License](#license)

## Introduction
This repository provides the code to empirically demonstrate how different models can learn to implement temporal difference (TD) methods for in-context policy evaluation. The experiments explore different models' ability to apply TD learning during inference without requiring parameter updates.

## Dependencies
To install the required dependencies, first clone this repository, then run the following command on the root directory of the project:
```bash
pip install .
```

## Usage

### Quick Start
To quickly replicate the experiments from Figure 2 of the paper, execute the following command:
```bash
python main.py --suffix=linear_standard -v
```
This will generate the following plots:
<p align="center">
  <img src="figs/P_metrics_1-1.png" alt="P Metrics Plot" height="150"/>
  <img src="figs/Q_metrics_1-1.png" alt="Q Metrics Plot" height="150"/>
  <img src="figs/PQ_mean_1_4000-1.png" alt="Final Learned P and Q" height="150"/>
  <img src="figs/cos_similarity-1.png" alt="Batch TD Comparison" height="150"/>
</p>
The generated figures will be saved in:

- `./logs/YYYY-MM-DD-HH-MM-SS/linear_standard/averaged_figures/` (aggregated results across all seeds)

- `./logs/YYYY-MM-DD-HH-MM-SS/linear_standard/seed_SEED/figures/` (diagnostic figures for each individual seed)

### Custom Experiment Settings
To run experiments with custom configurations, use:
```bash
python main.py [options]
```
Below is a list of the command-line arguments available for `main.py`:

- `-d`, `--dim_feature`: Feature dimension (default: 4)
- `-s`, `--num_states`: Number of states (default: 10)
- `-n`, `--context_length`: Context length (default: 30)
- `-l`, `--num_layers`: Number of layers (default: 3)
- `--gamma`: Discount factor (default: 0.9)
- `-mrp`, `--mrp_env`: MRP environment (choices: ['loop', 'boyan', 'cartpole', 'mountaincar'], default: 'boyan')
- `-config`, `--mrp_config`: Custom MRP presets (choices: ['none', 'demo_lp', 'demo_bc', 'boyan', 'cartpole', 'mountaincar'], default: 'none')
- `-model`, `--model_name`: Model type (choices: ['tf', 'mamba', 's4'], default='tf')
- `--mode`: Training mode auto-regressive or sequential (choices: ['auto', 'sequential'], default: 'auto')
- `--activation`: Activation function for transformers (choices: ['identity', 'softmax', 'relu'], default: 'identity')
- `--representable`: Flag to randomly sample a true weight vector that allows the value function to be fully represented by the features
- `--n_mrps`: Number of MRPs used for training (default: 4000)
- `--batch_size`: Mini-batch size (default: 64)
- `--n_batch_per_mrp`: Number of mini-batches sampled per MRP (default: 5)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Regularization term (default: 1e-6)
- `--loss`: Loss options (choices: ['mstde', 'msve_true', 'msve_mc'], default='mstde')
- `--log_interval`: Frequency of logging during training (default: 10)
- `--seed`: Random seeds (default: list(range(1, 31)))
- `--save_dir`: Directory to save logs (default: None)
- `--save_model`: Flag to save trained model for demo script
- `--suffix`: Suffix to append to the log save directory (default: None)
- `--gen_gif`: Flag to generate a GIF showing the evolution of weights (under construction)
- `--no_parallel`: Flag to disable multiprocessing
- `-v`, `--verbose`: Flag to print detailed training progress

Below is a list of configurations (set with `-config`)
- 'demo_lp': Generate a model in the 'loop' environment for demo script
  - `-mrp` = 'loop'
  - `-d` = 5
  - `-s` = 10
  - `-n` = 40
  - `--mode` = 'sequential'
  - `--representable`
  - `--n_mrps` = 4000
  - `--seed` = [0]
  - `--save_model`
  - `--no_parallel`
- 'demo_bc': Generate a model in the 'boyan' environment for demo script
  - `-mrp` = 'boyan'
  - `-d` = 4
  - `-s` = 10
  - `-n` = 40
  - `--mode` = 'sequential'
  - `--representable`
  - `--n_mrps` = 4000
  - `--seed` = [0]
  - `--save_model`
  - `--no_parallel`
- 'boyan': Configuration for 'boyan' environment
  - `-mrp` = 'boyan'
  - `-d` = 4
  - `-s` = 10
  - `-n` = 30
  - `--n_mrps` = 4000
- 'cartpole': Configuration for 'cartpole' environment
  - `-mrp` = 'cartpole'
  - `-d` = 4
  - `-s` = 81
  - `-n` = 100
  - `--n_mrps` = 5000
- 'mountaincar': Configuration for 'mountaincar' environment
  - `-mrp` = 'mountaincar'
  - `-d` = 2
  - `-s` = 25
  - `-n` = 100
  - `--n_mrps` = 5000

If no `--save_dir` is specified, logs will be saved in `./logs/YYYY-MM-DD-HH-MM-SS`. If a `--suffix` is provided, logs will be saved in `./logs/YYYY-MM-DD-HH-MM-SS/SUFFIX`.

### Demo
We have a demo script to demonstrate the performance of the TD algorithm implemented by our models under our theoretical construction.
The script generates a figure of the mean square value error (MSVE) averaged over the number of randomly generated MRPs against a sequence of increasing context lengths.
Note that we employ fully representable value functions here to make sure the minimum MSVE is zero.
<p align="center">
  <img src="figs/msve_vs_context_length.png" alt="Demo" height="150"/>
</p>

To run the script, use
```bash
python demo.py [options]
```
Below is a list of the command-line arguments available for `demo.py`:

- `-d`, `--dim_feature`: Feature dimension (default: 5)
- `-l`, `--num_layers`: Number of transformer layers/TD updates (default: 15)
- `-smin`, `--min_state_num`: Minimum possible state number of the randomly generated MRP (default: 5)
- `-smax`, `--max_state_num`: Maximum possible state number of the randomly generated MRP (default: 15)
- `--gamma`: Discount factor (default: 0.9)
- `-mrp`, `--mrp_env`: MRP environment (choices: ['loop', 'boyan'], default: 'loop')
- `-config`, `--mrp_config`: Custom MRP presets (choices: ['none', 'loop', 'boyan'], default: 'none')
- `-model`, `--model_name`: Model type(s) (choices: ['none', 'tf', 'tf_lin', 'mamba', 's4'], default=['none'])
- `-path`, `--model_path`: Path(s) to trained model (default=['none'])
- `--lr`: learning rate of the implemented in-context TD algorithm (default: 0.2)
- `--n_mrps`: Number of randomly generated MRPs to test on (default: 300)
- `-nmin`, `--min_ctxt_len`: Minimum context length (default: 1)
- `-nmax`, `--max_ctxt_len`: Maximum context length (default: 40)
- `--ctxt_step`: Context length increment step (default: 2)
- `--seed`: Random seed (default: 42)
- `--save_dir`: Directory to save demo results (default: 'logs')
- `--filename`: Filename of generated figure (default: 'msve_vs_context_length')

Below is a list of configurations (set with `-config`)
- 'loop': Configuration for 'loop' environment
  - `-mrp` = 'loop'
  - `-d` = 5
- 'boyan': Configuration for 'boyan' environment
  - `-mrp` = 'boyan'
  - `-d` = 4

By default, the result is saved to `./logs/demo`.

### Theory Verification
We provide a script to numerically verify our theories for transformers.
The script computes the absolute errors in log scale between the value predictions by the linear transformers and the direct implementations of their corresponding in-context algorithms.
<p align="center">
  <img src="figs/log_error.png" alt="Theory Verification" height="150"/>
</p>

To run the script, use
```bash
python verify.py [options]
```
Below is a list of the command-line arguments available for `verify.py`:
- `-d`, `--dim_feature`: Feature dimension (default: 3)
- `-n`, `--context_length`: Context length (default: 100)
- `-l`, `--num_layers`: Number of transformer layers/TD updates (default: 40)
- `--num_trials`: Number of trials to run (default: 30)
- `--seed`: Random seed (default: 42)
- `--save_dir`: Directory to save theory verification results (default: 'logs')

By default, the result is saved to `./logs/theory`.

### Complete Replication
To run all the experiments from the paper in one go, execute the following shell script:
```bash
./run.sh
```
We do not recommend running all the experiments for mamba at once, however all the experiments are contained in the following shell script:
```bash
./run_mamba.sh
```
### Test
To test run the experiments for transformers in small scale, execute the following shell script:
```bash
./test.sh
```
The test results are stored in `./logs/test`.
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
