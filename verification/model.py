import torch
import torch.nn as nn

from utils import stack_four

import sys
import numpy as np

sys.path.append('mamba')
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block as MambaBlock
sys.path.remove('mamba')

# TD layer


class DiscoutedTDLayer(nn.Module):
    def __init__(self, d: int, n: int, lmbd: float):
        '''
        d: feature dimension
        n: context length
        lmbd: eligibility trace decay
        '''
        super(DiscoutedTDLayer, self).__init__()
        self.d = d
        self.n = n
        self.P = torch.zeros((2 * d + 1, 2 * d + 1))
        self.P[-1, -1] = 1
        self.M = torch.eye(n + 1)
        for col in range(n):
            for row in range(col+1, n):
                self.M[row, col] = lmbd*self.M[row-1, col]
        self.M[-1, -1] = 0
        I = torch.eye(d)
        O = torch.zeros((d, d))
        self.M1 = stack_four(-I, I, O, O)
        self.C = torch.eye(d)
        self.B = stack_four(self.C.t(), O, O, O)
        self.A = torch.mm(self.B, self.M1)
        self.Q = torch.zeros_like(self.P)
        self.Q[:2*d, :2*d] = self.A

    def forward(self, Z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        '''
        Z: prompt matrix
        alpha: learning rate
        '''
        next_Z = Z + alpha / self.n * self.P @ Z @ self.M @ Z.T @ self.Q @ Z
        return next_Z


class DiscountedTDTransformer(nn.Module):
    def __init__(self, l: int, d: int, n: int, lmbd: float):
        '''
        l: number of layers
        d: feature dimension
        n: context length
        lmbd: eligibility trace decay
        '''
        super(DiscountedTDTransformer, self).__init__()
        self.n = n
        self.layers = nn.ModuleList(
            [DiscoutedTDLayer(d, n, lmbd) for _ in range(l)])
        # preconditioning matrices
        self.Cs = [layer.C for layer in self.layers]

    def forward(self, Z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        '''
        Z: prompt matrix
        alpha: learning rate
        '''
        v = []
        for layer in self.layers:
            Z = layer.forward(Z, alpha)
            v.append(-Z[-1, -1].item())
        v = torch.tensor(v)
        return v

# residual gradient layer


class RGLayer(nn.Module):
    def __init__(self, d: int, n: int):
        '''
        d: feature dimension
        n: context length
        '''
        super(RGLayer, self).__init__()
        self.d = d
        self.n = n
        self.P = torch.zeros((2 * d + 1, 2 * d + 1))
        self.P[-1, -1] = 1
        self.M = torch.eye(n + 1)
        self.M[-1, -1] = 0
        I = torch.eye(d)
        O = torch.zeros((d, d))
        self.M1 = stack_four(-I, I, O, O)
        self.M2 = -self.M1
        self.C = torch.eye(d)
        self.B = stack_four(self.C.t(), O, O, O)
        self.A = self.M2.t() @ self.B @ self.M1
        self.Q = torch.zeros_like(self.P)
        self.Q[:2*d, :2*d] = self.A

    def forward(self, Z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        next_Z = Z + alpha / self.n * self.P @ Z @ self.M @ Z.T @ self.Q @ Z
        return next_Z


class RGTransformer(nn.Module):
    def __init__(self, l: int, d: int, n: int):
        '''
        l: number of layers
        d: feature dimension
        n: context length
        '''
        super(RGTransformer, self).__init__()
        self.n = n
        self.layers = nn.ModuleList([RGLayer(d, n) for _ in range(l)])
        # preconditioning matrices
        self.Cs = [layer.C for layer in self.layers]

    def forward(self, Z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        '''
        Z: prompt matrix
        alpha: learning rate
        '''
        v = []
        for layer in self.layers:
            Z = layer.forward(Z, alpha)
            # negate it to align with the convention
            v.append(-Z[-1, -1].item())
        v = torch.tensor(v)
        return v

# averate reward TD layer


class AVGREWTDLayer(nn.Module):
    def __init__(self, d: int, n: int):
        '''
        d: feature dimension
        n: context length
        '''
        super(AVGREWTDLayer, self).__init__()
        self.d = d
        self.n = n

        self.P1 = torch.zeros((2 * d + 2, 2 * d + 2))
        self.P1[-2, -2] = 1  # head 1, filter out the reward row only
        self.P2 = torch.zeros((2 * d + 2, 2 * d + 2))
        # head 2, filter out the (cumulative) reward bar row only
        self.P2[-1, -1] = 1

        s = torch.ones((n + 1, n + 1))
        s = torch.triu(s)
        diag = torch.diag(torch.tensor([1/k for k in range(1, n + 2)]))
        self.R = s @ diag - torch.eye(n + 1)  # for computing r bar - r

        I = torch.eye(d)
        O = torch.zeros((d, d))
        self.M = torch.eye(n + 1)
        self.M[-1, -1] = 0
        self.M1 = stack_four(-I, I, O, O)
        self.C = torch.randn(d, d)
        self.B = stack_four(self.C.t(), O, O, O)
        self.A = torch.mm(self.B, self.M1)
        self.Q = torch.zeros_like(self.P1)
        self.Q[:2*d, :2*d] = self.A

        self.W = torch.zeros((2*d+2, 4*d + 4))
        self.W[-1, 2*d] = 1
        self.W[-1, -1] = 1

    def forward(self, Z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        head1 = self.P1 @ Z @ self.R @ self.M @ Z.T @ self.Q @ Z
        head2 = self.P2 @ Z @ self.M @ Z.T @ self.Q @ Z
        multihead = torch.concat([head1, head2], dim=0)

        next_Z = Z + alpha / self.n * self.W @ multihead
        return next_Z


class AVGREWTDTransformer(nn.Module):
    def __init__(self, l: int, d: int, n: int):
        super(AVGREWTDTransformer, self).__init__()
        self.n = n
        self.layers = nn.ModuleList([AVGREWTDLayer(d, n) for _ in range(l)])
        self.Cs = [layer.C for layer in self.layers]

    def forward(self, Z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        v = []
        for layer in self.layers:
            Z = layer.forward(Z, alpha)
            v.append(Z[-1, -1].item())
        v = torch.tensor(v)
        return v

# mamba

class MambaSSM(nn.Module):
    def __init__(self, l: int, d: int, mode: str = 'auto'):
        '''
        l: number of layers
        d: feature dimension
        '''
        super(MambaSSM, self).__init__()
        self.d = d
        self.l = l
        self.mode = mode
        self.Cs = [torch.eye(d) for _ in range(l)]
        if mode == 'auto':
            self.layer = MambaBlock(2*d+1, Mamba, nn.Identity)
        elif mode == 'sequential':
            self.layers = nn.ModuleList([
                Mamba(2*d+1, dtype=torch.float32)
            for i in range(l)])
            self.norms = nn.ModuleList([
                nn.LayerNorm(2*d+1, dtype=torch.float32)
            for _ in range(l)])
            # self.layers = nn.ModuleList([
            #     MambaBlock(2*d+1, Mamba, nn.Identity)
            # for _ in range(l)])
        elif mode == 'standalone':
            self.layer = Mamba(2*d+1, dtype=torch.float32)
            self.Cs = [torch.eye(d)]
        else:
            raise ValueError('mode must be auto, sequential, or standalone')

    def forward(self, Z) -> torch.Tensor:
        v = []
        Z = Z.float()
        residual = None
        if self.mode == 'auto':
            for _ in range(self.l):
                Z.transpose_(0, 1)
                Z.unsqueeze_(0)
                Z, residual = self.layer(Z.to(self.device), residual, dtype=torch.float32, 
                                         device=self.device).to('cpu')
                Z = (Z + residual) if residual is not None else Z
                Z.squeeze_(0)
                Z.transpose_(0, 1)
                v.append(Z[-1, -1].item())
        elif self.mode == 'sequential':
            for layer, norm in zip(self.layers, self.norms):
                Z.transpose_(0, 1)
                Z.unsqueeze_(0)
                residual = Z
                Z = layer(Z)
                Z = Z + residual
                Z = layer(Z)
                Z = norm(Z)
                Z.squeeze_(0)
                Z.transpose_(0, 1)
                v.append(Z[-1, -1].item())
        else:
            Z.transpose_(0, 1)
            Z.unsqueeze_(0)
            Z = Z.float()
            Z = self.layer(Z.to(self.device))
            Z.squeeze_(0)
            Z.transpose_(0, 1)
            v.append(Z[-1, -1].item())
        return torch.tensor(v)