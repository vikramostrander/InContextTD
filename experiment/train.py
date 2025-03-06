import datetime
import json
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from experiment.model import HardLinearTransformer, Transformer, MambaSSM, S4SSM
from experiment.prompt import MRPPromptGenerator
from MRP.mrp import MRP
from utils import (set_seed, compute_msve, cos_sim, solve_msve_weight)


def _init_log() -> dict:
    log = {'xs': [],
           'alpha': [],
           'v_model v_td msve': [],
           'implicit_weight_sim': [],
           'sensitivity cos sim': [],
           'P': [],
           'Q': []}
    return log


def _init_save_dir(save_dir: str) -> None:
    if save_dir is None:
        startTime = datetime.datetime.now()
        save_dir = os.path.join('./logs',
                                "train",
                                startTime.strftime("%Y-%m-%d-%H-%M-%S"))

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def _save_log(log: dict, save_dir: str) -> None:
    for key, value in log.items():
        log[key] = np.array(value)
    np.savez(os.path.join(save_dir, 'data.npz'), **log)


def compare_sensitivity(model,
                        batch_td,
                        prompt,
                        device):
    '''
    computes the expected cosine similarity and l2 norm
    between the models' gradients w.r.t query
    '''
    prompt = prompt.copy()
    Phi: torch.Tensor = prompt.get_feature_mat()
    steady_d: np.ndarray = prompt.mrp.steady_d
    mean_cos_sim = 0.0
    mean_l2_dist = 0.0
    for s, feature in enumerate(Phi):
        prompt.set_query(feature)
        prompt.enable_query_grad()

        model_v = model.pred_v(prompt.z().to(device))
        model_v.backward()
        model_grad = prompt.query_grad().cpu().numpy()
        prompt.zero_query_grad()

        td_v = batch_td.pred_v(prompt.z())
        td_v.backward()
        td_grad = prompt.query_grad().numpy()
        prompt.disable_query_grad()

        mean_cos_sim += steady_d[s]*cos_sim(model_grad, td_grad)
    return mean_cos_sim


def implicit_weight_sim(v_model: np.ndarray,
                        batch_td,
                        prompt):
    '''
    computes the cosine similarity and l2 distance
    between the batch TD weight (with the fitted learning rate) 
    and the weight of the best linear model that explaines v_model
    '''
    prompt = prompt.copy()
    steady_d = prompt.mrp.steady_d
    Phi = prompt.get_feature_mat().numpy()
    w_model = solve_msve_weight(steady_d, Phi, v_model).flatten()
    prompt.enable_query_grad()
    v_td = batch_td.pred_v(prompt.z())
    v_td.backward()
    w_td = prompt.query_grad().numpy().flatten()
    prompt.zero_query_grad()
    prompt.disable_query_grad()

    return cos_sim(w_model, w_td)


def train(d: int,
          s: int,
          n: int,
          l: int,
          gamma: float = 0.9,
          mrp_class: str = 'boyan',
          model_name: str = 'tf',
          mode: str = 'auto',
          activation: str = 'identity',
          norm: str = 'none',
          sample_weight: bool = False,
          lr: float = 0.001,
          weight_decay=1e-6,
          n_mrps: int = 1000,
          mini_batch_size: int = 64,
          n_batch_per_mrp: int = 5,
          log_interval: int = 10,
          save_dir: str = None,
          save_model: bool = False,
          random_seed: int = 2) -> None:
    '''
    d: feature dimension
    s: number of states
    n: context length
    l: number of layers
    gamma: discount factor
    mrp_class: type of MRP environment (e.g. boyan, lake, cartpole)
    model_name: type of model (e.g. tf, mamba, s4)
    mode: 'auto', 'sequential', or 'standalone'
    activation: activation function (e.g. softmax, identity, relu)
    norm: normalization function for mamba (none or layer)
    sample_weight: sample a random true weight vector
    lr: learning rate
    weight_decay: regularization
    n_mrps: number of MRPs
    mini_batch_size: mini batch size
    n_batch_per_mrp: number of batches per MRP
    log_interval: logging interval
    save_dir: directory to save logs
    save_model: save model weights
    random_seed: random seed
    '''

    _init_save_dir(save_dir)

    set_seed(random_seed)

    if model_name == 'mamba':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = MambaSSM(d, l, device=device, norm=norm, mode=mode).to(device)
        else:
            raise Exception("error: cuda not found, required for mamba")
    # elif model_name == 's4':
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # model = S4SSM(d, l, activation=activation, mode=mode).to(device)
    else:
        device = torch.device('cpu')
        model = Transformer(d, n, l, activation=activation, mode=mode)

    # this is the hardcoded transformer that implements Batch TD with fixed weights
    batch_td = HardLinearTransformer(d, n, l)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_hard = optim.Adam(batch_td.parameters(),
                          lr=lr, weight_decay=weight_decay)
    log = _init_log()

    pro_gen = MRPPromptGenerator(s, d, n, gamma, mrp_class)

    ### Training Loop ###
    for i in tqdm(range(1, n_mrps+1)):
        pro_gen.reset_feat()  # reset feature
        pro_gen.reset_mrp(sample_weight=sample_weight)  # reset MRP
        prompt = pro_gen.get_prompt()  # get prompt object
        for _ in range(n_batch_per_mrp):
            mstde = 0.0
            mstde_hard = 0.0
            Z_0 = prompt.reset()
            v_current = model.pred_v(Z_0.to(device)).cpu()
            v_hard_current = batch_td.pred_v(Z_0)
            for _ in range(mini_batch_size):
                Z_next, reward = prompt.step()  # slide window
                v_next = model.pred_v(Z_next.to(device)).cpu()
                v_hard_next = batch_td.pred_v(Z_next)
                tde = reward + gamma*v_next.detach() - v_current 
                tde_hard = reward + gamma*v_hard_next.detach() - v_hard_current
                mstde += tde**2
                mstde_hard += tde_hard**2
                v_current = v_next
                v_hard_current = v_hard_next
            mstde /= mini_batch_size  # MSTDE for the trainable transformer
            mstde_hard /= mini_batch_size  # MSTDE for the hardcoded transformer
            opt.zero_grad()
            mstde.backward()
            opt.step()
            # the learning rate for batch td (alpha) is still trainable so we need to backpropagate
            opt_hard.zero_grad()
            mstde_hard.backward()
            opt_hard.step()

        if i % log_interval == 0:
            prompt.reset()  # reset prompt for fair testing
            mrp: MRP = prompt.mrp

            phi: np.ndarray = prompt.get_feature_mat().numpy()
            steady_d: np.ndarray = mrp.steady_d

            v_model: np.ndarray = model.fit_value_func(
                prompt.context().to(device), torch.from_numpy(phi).to(device)
            ).detach().cpu().numpy()
            v_td: np.ndarray = batch_td.fit_value_func(
                prompt.context(), torch.from_numpy(phi)
            ).detach().numpy()

            log['xs'].append(i)
            log['alpha'].append(batch_td.attn.alpha.item())

            # Value Difference (VD)
            log['v_model v_td msve'].append(compute_msve(v_model, v_td, steady_d))

            # Sensitivity Similarity (SS)
            sens_cos_sim = compare_sensitivity(model, batch_td, prompt, device)
            log['sensitivity cos sim'].append(sens_cos_sim)

            # Implicit Weight Similarity (IWS)
            iws = implicit_weight_sim(v_model, batch_td, prompt)
            log['implicit_weight_sim'].append(iws)

            if model_name == 'tf':
                if mode == 'auto':
                    log['P'].append([model.attn.P.detach().numpy().copy()])
                    log['Q'].append([model.attn.Q.detach().numpy().copy()])
                else:
                    log['P'].append(
                        np.stack([layer.P.detach().numpy().copy() for layer in model.layers]))
                    log['Q'].append(
                        np.stack([layer.Q.detach().numpy().copy() for layer in model.layers]))

    _save_log(log, save_dir)

    hyperparameters = {
        'd': d,
        's': s,
        'n': n,
        'l': l,
        'gamma': gamma,
        'mrp_class': mrp_class,
        'model': model_name,
        'mode': mode,
        'activation': activation,
        'sample_weight': sample_weight,
        'n_mrps': n_mrps,
        'mini_batch_size': mini_batch_size,
        'n_batch_per_mrp': n_batch_per_mrp,
        'lr': lr,
        'weight_decay': weight_decay,
        'log_interval': log_interval,
        'random_seed': random_seed,
        'linear': True if activation == 'identity' else False,
    }

    # Save hyperparameters as JSON
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(hyperparameters, f)

    # Save model weights
    if save_model:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_state_dict.pth'))