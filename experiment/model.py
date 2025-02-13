import torch
import torch.nn as nn

import sys

from experiment.mamba import Mamba
sys.path.append('mamba')
# from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block as MambaBlock
sys.path.remove('mamba')

sys.path.append('s4')
from src.models.sequence.modules.s4block import S4Block
sys.path.remove('s4')

from utils import stack_four


class Attention(nn.Module):
    def __init__(self,
                 d: int,
                 n: int,
                 activation: str):
        '''
        d: feature dimension
        n: context length
        actvation: activation function
        '''
        super(Attention, self).__init__()
        self.d = d
        self.n = n
        self.activation = get_activation(activation)

        M = torch.eye(n + 1)
        M[-1, -1] = 0
        self.M = M

        self.P = nn.Parameter(torch.empty(2 * d + 1, 2 * d + 1))
        self.Q = nn.Parameter(torch.empty(2 * d + 1, 2 * d + 1))

    def forward(self, Z):
        X = Z.T @ self.Q @ Z
        return Z + 1.0 / self.n * self.P @ Z @ self.M @ self.activation(X)


class Transformer(nn.Module):
    def __init__(self,
                 d: int,
                 n: int,
                 l: int,
                 activation: str,
                 mode='auto'):
        '''
        d: feature dimension
        n: context length
        l: number of layers
        activation: activation function (e.g. identity, softmax, identity, relu)
        mode: 'auto' or 'sequential'
        '''
        super(Transformer, self).__init__()
        self.d = d
        self.n = n
        self.l = l
        self.mode = mode
        if mode == 'auto':
            attn = Attention(d, n, activation)
            nn.init.xavier_normal_(attn.P, gain=0.1)
            nn.init.xavier_normal_(attn.Q, gain=0.1)
            self.attn = attn
        elif mode == 'sequential':
            self.layers = nn.ModuleList(
                [Attention(d, n, activation) for _ in range(l)])
            for attn in self.layers:
                nn.init.xavier_normal_(attn.P, gain=0.1/l)
                nn.init.xavier_normal_(attn.Q, gain=0.1/l)
        else:
            raise ValueError('mode must be either auto or sequential')

    def forward(self, Z):
        '''
        Z: prompt of shape (2*d+1, n+1)
        '''
        if self.mode == 'auto':
            for _ in range(self.l):
                Z = self.attn(Z)
        else:
            for attn in self.layers:
                Z = attn(Z)
        return Z

    def fit_value_func(self,
                       context: torch.Tensor,
                       phi: torch.Tensor) -> torch.Tensor:
        '''
        context: the context of shape
        phi: features of shape (s, d)
        returns the fitted value function given the context in shape (s, 1)
        '''
        v_vec = []
        for feature in phi:
            feature_col = torch.zeros((2 * self.d + 1, 1))
            feature_col[:self.d, 0] = feature
            Z_p = torch.cat([context, feature_col], dim=1)
            v = self.pred_v(Z_p)
            v_vec.append(v)
        tf_v = torch.stack(v_vec, dim=0).unsqueeze(1)
        return tf_v

    def pred_v(self, Z: torch.Tensor) -> torch.Tensor:
        '''
        Z: prompt of shape (2*d+1, n+1)
        predict the value of the query feature
        '''
        Z_tf = self.forward(Z)
        return -Z_tf[-1, -1]


class HardLinearAttention(nn.Module):
    def __init__(self, d: int, n: int):
        '''
        d: feature dimension
        n: context length
        '''
        super(HardLinearAttention, self).__init__()

        self.d = d
        self.n = n
        self.alpha = nn.Parameter(torch.ones(1))
        P = torch.zeros((2 * d + 1, 2 * d + 1))
        P[-1, -1] = 1.0
        self.P = P

        M = torch.eye(n+1)
        M[-1, -1] = 0
        self.M = M

        I = torch.eye(d)
        O = torch.zeros((d, d))
        A = stack_four(-I, I, O, O)
        Q = torch.zeros((2 * d + 1, 2 * d + 1))
        Q[:2 * d, :2 * d] = A
        self.Q = Q

    def forward(self, Z):
        return Z + self.alpha / self.n * self.P @ Z @ self.M @ Z.T @ self.Q @ Z


class HardLinearTransformer(nn.Module):
    def __init__(self,
                 d: int,
                 n: int,
                 l: int):
        '''
        d: feature dimension
        n: context length
        l: number of layers
        '''
        super(HardLinearTransformer, self).__init__()
        self.d = d
        self.n = n
        self.l = l
        self.attn = HardLinearAttention(d, n)

    def forward(self, Z):
        '''
        Z: prompt of shape (2*d+1, n+1)
        '''
        for _ in range(self.l):
            Z = self.attn(Z)

        return Z

    def fit_value_func(self,
                       context: torch.Tensor,
                       phi: torch.Tensor) -> torch.Tensor:
        '''
        context: the context of shape (2*d+1, n)
        phi: features of shape (s, d)
        returns the fitted value function given the context in shape (s, 1)
        '''
        tf_v = []
        for feature in phi:
            feature_col = torch.zeros((2*self.d+1, 1))
            feature_col[:self.d, 0] = feature
            Z_p = torch.cat([context, feature_col], dim=1)
            v_tf = self.pred_v(Z_p)
            tf_v.append(v_tf)
        tf_v = torch.stack(tf_v, dim=0).unsqueeze(1)
        return tf_v

    def pred_v(self, Z: torch.Tensor) -> torch.Tensor:
        '''
        Z: prompt of shape (2*d+1, n+1)
        predict the value of the query feature
        '''
        Z_tf = self.forward(Z)
        return -Z_tf[-1, -1]


class MambaSSM(nn.Module):
    def __init__(self,
                 d: int,
                 l: int,
                 activation: str,
                 device: torch.device,
                 mode='auto'):
        super(MambaSSM, self).__init__()
        self.d = d
        self.l = l
        self.device = device
        self.mode = mode
        assert activation in {'identity', 'silu'}
        if mode == 'auto':
            self.layer = Mamba(2*d+1, activation=activation, device=self.device)
        elif mode == 'sequential':
            self.layers = nn.ModuleList([
                Mamba(2*d+1, activation=activation, device=self.device)
            for _ in range(l)])
        elif mode == 'standalone':
            self.layer = Mamba(2*d+1, activation=activation, device=self.device)
        else:
            raise ValueError('mode must be auto, sequential, or standalone')

    def forward(self, Z):
        Z.transpose_(0, 1)
        Z.unsqueeze_(0)
        if self.mode == 'auto':
            for _ in range(self.l):
                residual = Z
                Z = self.layer(Z)
                Z += residual
        elif self.mode == 'sequential':
            for layer in self.layers:
                residual = Z
                Z = layer(Z)
                Z += residual
        else:
            Z = self.layer(Z)
        Z.squeeze_(0)
        Z.transpose_(0, 1)
        return Z
    
    def fit_value_func(self,
                       context: torch.Tensor,
                       phi: torch.Tensor):
        v_vec = []
        for feature in phi:
            feature_col = torch.zeros((2 * self.d + 1, 1), device=self.device)
            feature_col[:self.d, 0] = feature
            Z_p = torch.cat([context, feature_col], dim=1)
            v = self.pred_v(Z_p)
            v_vec.append(v)
        mamba_v = torch.stack(v_vec, dim=0).unsqueeze(1)
        return mamba_v
    
    def pred_v(self, Z):
        Z_mamba = self.forward(Z.clone())
        return Z_mamba[-1, -1]


class S4SSM(nn.Module):
    def __init__(self,
                 d: int,
                 l: int,
                 activation='gelu',
                 mode='auto'):
        super(S4SSM, self).__init__()
        self.d = d
        self.l = l
        self.mode = mode
        assert activation in {'gelu', 'identity', 'elu', 'tanh', 'relu'}
        if mode == 'auto':
            self.layer = S4Block(2*d+1, activation=activation)
            self.norm = nn.LayerNorm(2*d+1)
        elif mode == 'sequential':
            self.layers = nn.ModuleList([
                S4Block(2*d+1, activation=activation)
            for _ in range(l)])
            self.norms = nn.ModuleList([
                nn.LayerNorm(2*d+1) 
            for _ in range(l)])
        elif mode == 'standalone':
            self.layer = S4Block(2*d+1, activation=activation)
        else:
            raise ValueError('mode must be either auto or sequential')

    def forward(self, Z):
        Z.unsqueeze_(0)
        if self.mode == 'auto':
            for _ in range(self.l):
                residual = Z
                Z, _ = self.layer(Z)
                Z = Z + residual
                Z = self.norm(Z.transpose(-1, -2)).transpose(-1, -2)
        elif self.mode == 'sequential':
            for layer, norm in zip(self.layers, self.norms):
                residual = Z
                Z, _ = layer(Z)
                Z = Z + residual
                Z = norm(Z.transpose(-1, -2)).transpose(-1, -2)
        else:
            Z, _ = self.layer(Z)
        Z.squeeze_(0)
        return Z
    
    def fit_value_func(self,
                       context: torch.Tensor,
                       phi: torch.Tensor):
        v_vec = []
        for feature in phi:
            feature_col = torch.zeros((2 * self.d + 1, 1), device=phi.get_device())
            feature_col[:self.d, 0] = feature
            Z_p = torch.cat([context, feature_col], dim=1)
            v = self.pred_v(Z_p)
            v_vec.append(v)
        s4_v = torch.stack(v_vec, dim=0).unsqueeze(1)
        return s4_v
    
    def pred_v(self, Z):
        Z_s4 = self.forward(Z.clone())
        return Z_s4[-1][-1]


def get_activation(activation: str) -> nn.Module:
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'softmax':
        return nn.Softmax(dim=1)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Invalid activation function: {activation}")
