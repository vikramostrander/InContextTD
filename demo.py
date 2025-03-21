import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
from tqdm import tqdm

from experiment.prompt import Feature, MRPPrompt
from experiment.model import Transformer, S4SSM
from MRP.loop import Loop
from MRP.boyan import BoyanChain
from utils import compute_msve, set_seed


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--dim_feature', type=int,
                        help='feature dimension', default=5)
    parser.add_argument('-l', '--num_layers', type=int,
                        help='number of transformer layers', default=15)
    parser.add_argument('-smin', '--min_state_num', type=int,
                        help='minimum possible number of states', default=5)
    parser.add_argument('-smax', '--max_state_num', type=int,
                        help='maximum possible number of states', default=15)
    parser.add_argument('-mrp', '--mrp_env', type=str,
                        help='MRP environment', default='loop', 
                        choices=['loop', 'boyan'])
    parser.add_argument('-config', '--mrp_config', type=str,
                        help='custom MRP presets', default='none', 
                        choices=['none', 'loop', 'boyan'])
    parser.add_argument('-model', '--model_name', type=str, nargs='+',
                        help='model type(s)', default=['none'], choices=['tf', 'mamba', 's4'])
    parser.add_argument('--activation', type=str,
                        help='activation function for transformers', default='identity')
    parser.add_argument('-path', '--model_path', type=str, nargs='+',
                        help='path to trained model', default=['none'])
    parser.add_argument('--gamma', type=float,
                        help='discount factor', default=0.9)
    parser.add_argument('--lr', type=float,
                        help='TD learning rate', default=0.2)
    parser.add_argument('--n_mrps', type=int,
                        help='total number of MRPs to run', default=300)
    parser.add_argument('-nmin', '--min_ctxt_len', type=int,
                        help='minimum context length', default=1)
    parser.add_argument('-nmax', '--max_ctxt_len', type=int,
                        help='maximum context length', default=40)
    parser.add_argument('--ctxt_step', type=int,
                        help='context length step', default=2)
    parser.add_argument('--seed', type=int,
                        help='random seed', default=42)
    parser.add_argument('--save_dir', type=str,
                        help='directory to save demo result', default='logs')

    args: Namespace = parser.parse_args()

    save_path = os.path.join(args.save_dir, 'demo')
 
    os.makedirs(save_path, exist_ok=True)

    set_seed(args.seed)

    d = args.dim_feature
    l = args.num_layers
    min_s = args.min_state_num
    max_s = args.max_state_num
    gamma = args.gamma
    n_mrps = args.n_mrps
    alpha = args.lr
    context_lengths = list(range(args.min_ctxt_len,
                                 args.max_ctxt_len+1,
                                 args.ctxt_step))
    
    if args.mrp_config != 'none':
        if args.mrp_config == 'loop':
            args.mrp_env = 'loop'
            d = 5
        if args.mrp_config == 'boyan':
            args.mrp_env = 'boyan'
            d = 4

    assert len(args.model_name) == len(args.model_path)
    results = dict()
    for model_name, model_path in zip(args.model_name, args.model_path):
        if model_name == 'mamba':
            if not torch.cuda.is_available():
                raise Exception("error: cuda not found, required for mamba")
            device = torch.device('cuda')
            if not model_path:
                raise Exception("error: trained model required for mamba")
            model = torch.load(model_path)
        elif model_name == 's4':
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            if not model_path:
                raise Exception("error: trained model required for s4")
            model = S4SSM(d, l, device=device, mode='sequential').to(device)
            model.load_state_dict(torch.load(model_path))
        elif model_name == 'tf':
            device = torch.device('cpu')
            model = Transformer(d, args.max_ctxt_len, l, activation=args.activation, mode='sequential')
            model.load_state_dict(torch.load(model_path))
        else:
            model = None    # defaults to td update

        all_msves = []  # (n_mrps, len(context_lengths))
        for _ in tqdm(range(n_mrps)):
            s = np.random.randint(min_s, max_s + 1)  # sample number of states
            thd = np.random.uniform(low=0.1, high=0.9)
            feature = Feature(d, s)  # new feature
            true_w = np.random.randn(d, 1)  # sample true weight
            if args.mrp_env == 'loop':
                mrp = Loop(s, gamma, threshold=thd, weight=true_w, phi=feature.phi)
            elif args.mrp_env == 'boyan':
                mrp = BoyanChain(s, weight=true_w, X=feature.phi)
            msve_n = []
            for n in context_lengths:
                prompt = MRPPrompt(d, n, gamma, mrp, feature)
                prompt.reset()
                if model is not None:
                    ctxt = prompt.context()
                    if model_name == 'tf':
                        ctxt = torch.tensor(np.pad(ctxt, ((0,0),(args.max_ctxt_len-n,0)), 
                                                   mode='constant', constant_values=0))
                    v = model.fit_value_func(
                        ctxt.to(device), 
                        torch.from_numpy(feature.phi).to(device)
                    ).detach().cpu().numpy()
                else:
                    w = torch.zeros((d, 1))
                    for _ in range(l):
                        w, _ = prompt.td_update(w, lr=alpha)
                    v = feature.phi @ w.numpy()
                msve = compute_msve(v, mrp.v, mrp.steady_d)
                msve_n.append(msve)
            all_msves.append(msve_n)

        all_msves = np.array(all_msves)
        mean = np.mean(all_msves, axis=0)
        ste = np.std(all_msves, axis=0) / np.sqrt(n_mrps)

        results[model_name] = mean, ste

    plt.style.use(['science', 'bright', 'no-latex'])
    fig = plt.figure()
    colors = {'tf': 'b', 'mamba': 'r', 's4': 'g'}
    for model_name in args.model_name:
        mean, ste = results[model_name]
        plt.plot(context_lengths, mean, 
                 label=model_name, color=colors[model_name])
        plt.fill_between(context_lengths,
                         np.clip(mean - ste, a_min=0, a_max=None),
                         mean + ste,
                         color=colors[model_name], alpha=0.2)
    plt.xlabel('Context Length (t)')
    plt.ylabel('MSVE', rotation=0, labelpad=30)
    plt.legend()
    plt.ylim(ymin=0)
    plt.grid(True)
    fig_path = os.path.join(save_path, 'msve_vs_context_length.pdf')
    plt.savefig(fig_path, dpi=300, format='pdf')
    plt.close(fig)
