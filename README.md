# MCCFVFP

## OverView

This is the CFVFP method corresponding to the paper titled "Accelerating Nash Equilibrium Convergence in Monte Carlo Settings Through Counterfactual Value Based Fictitious Play" [Link](https://arxiv.org/abs/2309.03084). This paper has been accepted by NeurIPS 2024.

This method is employed to address large-scale incomplete information game problems. The core concept of this approach is to use Q values instead of regret values in CFR. In the next iteration, directly utilize the strategy with the max Q-value action as the current strategy. This not only sidesteps a great deal of cumbersome calculations related to regret values but also can prevent the selection of dominated strategies, thereby accelerating the convergence of game algorithms.

For the experiments, we have adopted common benchmark experimental environments such as Kuhn and Leduc. Additionally, our algorithm has been successfully applied in Texas Hold'em. https://gtoking.com is a commercial solver based on MCCFVFP. If necessary, you can try it out.

## Instructions

```python
game_name = 'Leduc'
is_show_policy = False
prior_state_num = 3
y_pot = 3
z_len = 3
```

The above code is used to set the experimental configuration, where 'game_name' determines the type of game to be trained. This can be referenced from the code below or from the 'Game_Sampling'.

The 'prior_state_num' is used to set the scale of the game, and there are several ways to understand it: For 'Kuhn', 'Leduc', 'KuhnNPot', 'Leduc3Pot', and 'Leduc5Pot', it can be understood as the number of cards set in the game. For 'Goofspiel', it can be understood as the number of cards in each player’s hand. For 'PAM', it can be understood as the step length to terminate the game.

The 'y_pot''z_len' is only for when the 'game_name' is 'KuhnNPot'.

```python
train_mode = 'fix_itr'
log_interval_mode = 'itr'
log_mode = 'exponential'
```

The 'train_mode' is set to the mode used for training: For "train_mode = 'fix_itr'"it means fix number of training rounds. For "train_mode = 'node_touched'" it means fix number of nodes passed during training. For "train_mode = 'tran_time'"it means fix training time.

The 'log_interval_mode' is set to the mode used for recording results.

The 'log_mode' is set to the interval for recording results: For'exponential' records in exponential form. Fpr 'normal' records in arithmetic form.

```python
total_train_constraint = 1000000
log_interval = 1.5
nun_of_train_repetitions = 3
n_jobs = 1  
```

The setting of 'total_train_constraint' depends on the selected 'train_mode': For "train_mode = 'tran_time'" then the 'total_train_constraint=100' means that each method is trained for 100s. For "train_mode = 'node_touched'" means that each method is trained to pass a set number of nodes.

The 'log_interval' is used to set the recording of training results.

The 'nun_of_train_repetitions' is used to set how many times a training is repeated.

The 'n_jobs' is for parallel training, generally 1 for your own compute.

After setting the above parameters, you can start running the experiment you want.