# esn_ctf

This directory contains implementations of echo state networks [1,2] (ESNs) for forecasting dynamical systems, as well as supporting evaluation in the Common Task Framework (CTF) for Science. As implemented [3, 4], an ESN is a form of recurrent neural that evolves according to 
$$h(t+1) = \alpha \tanh (W_{r}h(t) + W_{in} u(t) + \sigma_b \mathbf 1) + (1-\alpha)h(t),$$
where $h$ is the latent state, $\alpha$ is a leak rate, $\sigma_b$ is a scalar bias term, and $W_r$ and $W_{in}$ are network parameters. In contrast to other forms of RNN, $W_r$ and $W_{in}$ are randomly initialized and left fixed. A linear map, $W_{out}$, is then learned via ridge regression such that $W_{out} g(h(t)) \approx u(t)$ where $g: \mathbb R^h \to \mathbb R^h$. Often, $g$ is taken to be the identity mapping, but other choices are sometimes necessary [4]. The cost of this simple training procedure is that $h$ must be taken to be high-dimensional. 

Within the CTF, we evaluate the performance of ESNs on low-dimensional systems (i.e. ODE systems) as presented in [3] and on high-dimensional systems as presented in [5].

## Usage
First, ensure that `ctf4science` is already installed. Then, from the root directory of `esn_ctf`, 
```bash
pip install -r requirements.txt
```
will install the remaining dependencies. 

Given a config file (examples in config directory), one can train and evaluate a model by running
```bash
python run.py <path-to-config>
```
Or, given a hyperparameter tuning config (examples in tuning_config directory), one can optimize a set of hyperparameters by running
```bash
python optimize_parameters.py --config_path <path-to-config>
```
The bash scripts `driver_run.sh` and `test_opt_params.sh` can provide a comprehensive test that your environment has been set up correctly by replicating the results fronm the CTF paper. It is strongly recommended to update the tuning config files to enforce a maximum number of trials before running `test_opt_params.sh`. 

## Citations
[1] Jaeger, "The echo state approach to analyzing and training recurrent neural networks," *German National Research Center for Information Technology, Technical Report GMD 148*, 2001.

[2] Maas and Markram, "On the computational power of circuits of spiking neurons," *Journal of Computer and System Sciences*, 2004.

[3] Platt et al., "A systematic exploration of reservoir computing for forecasting complex spatiotemporal dynamics," *Neural Networks*, 2022.

[4] Kidger and Garcia, "Equinox: neural networks in (JAX) via callable PyTrees and filtered transformations," *Differentiable Programming workshop at Neural Information Processing Systems*, 2021.

[5] Pathak et al., "Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach," *Physical Review Letters*, 2018.