# =============================================================================
# Intelligent Hospital Resource Management System
# Algorithm: ACPL (Adaptive Consequence-Penalised Learning) v2
#
# Copyright (c) 2025 Shashank Dev. All rights reserved.
# ACPL is an original algorithm developed by the author.
# Unauthorised reproduction of the ACPL core is prohibited.
# =============================================================================

import csv
import datetime
import json
import logging
import logging.handlers
import math
import os
import pathlib
import signal
import sys
import time
import uuid
import warnings
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque, Set
import numpy as np
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, lfilter

_norm2 = lambda v: math.hypot(float(v[0]), float(v[1]))

def _acpl_relu(x):    return np.maximum(0.0, x)
def _acpl_sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
def _acpl_tanh(x):    return np.tanh(np.clip(x, -20, 20))
def _acpl_d_relu(x):    return (x > 0).astype(np.float32)
def _acpl_d_sigmoid(s): return s * (1.0 - s)
def _acpl_d_tanh(t):    return 1.0 - t**2


class _ACPLAdam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m  = [np.zeros_like(p) for p in params]
        self.v  = [np.zeros_like(p) for p in params]
        self.t  = 0

    def step(self, grads):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1.0 - self.b1) * g
            v[:] = self.b2 * v + (1.0 - self.b2) * g * g
            p   -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)


class _ACPLLinear:
    def __init__(self, in_dim, out_dim, rng, scale=None):
        s = scale or np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, s, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, np.float32)
        self._x = None

    def forward(self, x):
        self._x = np.asarray(x, np.float32)
        return self._x @ self.W + self.b

    def backward(self, d_out):
        dW = self._x.T @ d_out
        db = d_out.sum(0)
        dx = d_out @ self.W.T
        return dx, dW, db

    def params(self): return [self.W, self.b]


class _ACPLGRU:
    def __init__(self, input_dim, hidden_dim, rng):
        s = np.sqrt(1.0 / hidden_dim)
        def _w(r, c): return rng.normal(0, s, (r, c)).astype(np.float32)
        H = hidden_dim; I = input_dim
        self.Wz = _w(I, H); self.Uz = _w(H, H); self.bz = np.zeros(H, np.float32)
        self.Wr = _w(I, H); self.Ur = _w(H, H); self.br = np.zeros(H, np.float32)
        self.Wh = _w(I, H); self.Uh = _w(H, H); self.bh = np.zeros(H, np.float32)
        self.hidden_dim = hidden_dim

    def forward(self, x, h):
        z = _acpl_sigmoid(x @ self.Wz + h @ self.Uz + self.bz)
        r = _acpl_sigmoid(x @ self.Wr + h @ self.Ur + self.br)
        g = _acpl_tanh(x   @ self.Wh + (r * h) @ self.Uh + self.bh)
        return (1.0 - z) * h + z * g

    def zero_state(self, batch=1):
        return np.zeros((batch, self.hidden_dim), np.float32)

    def params(self):
        return [self.Wz, self.Uz, self.bz,
                self.Wr, self.Ur, self.br,
                self.Wh, self.Uh, self.bh]


class _ACPLMiniNet:
    def __init__(self, in_dim, out_dim, hidden_dim, rng, lr=3e-4):
        self.l1 = _ACPLLinear(in_dim,    hidden_dim, rng)
        self.l2 = _ACPLLinear(hidden_dim, hidden_dim, rng)
        self.l3 = _ACPLLinear(hidden_dim, out_dim,   rng)
        all_p = self.l1.params() + self.l2.params() + self.l3.params()
        self.opt = _ACPLAdam(all_p, lr=lr)
        self._cache = {}

    def forward(self, x):
        x = np.asarray(x, np.float32)
        a1 = _acpl_relu(self.l1.forward(x))
        a2 = _acpl_relu(self.l2.forward(a1))
        out = self.l3.forward(a2)
        self._cache = {"x": x, "a1": a1, "a2": a2}
        return out

    def backward_update(self, d_out):
        c  = self._cache
        dx3, dW3, db3 = self.l3.backward(d_out)
        dx2 = dx3 * _acpl_d_relu(c["a2"])
        dx2r, dW2, db2 = self.l2.backward(dx2)
        dx1 = dx2r * _acpl_d_relu(c["a1"])
        _, dW1, db1 = self.l1.backward(dx1)
        self.opt.step([dW1, db1, dW2, db2, dW3, db3])

    def params_flat(self):
        return self.l1.params() + self.l2.params() + self.l3.params()


class _ACPLStateNorm:
    def __init__(self, dim, eps=1e-8, clip=5.0):
        self.mean  = np.zeros(dim, np.float64)
        self.var   = np.ones(dim, np.float64)
        self.count = 0
        self.eps   = eps
        self.clip  = clip

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(1, -1)
        n = x.shape[0]
        delta     = x.mean(0) - self.mean
        total     = self.count + n
        self.mean = self.mean + delta * n / total
        self.var  = (self.var * self.count + x.var(0) * n +
                     delta**2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x):
        x = np.asarray(x, np.float32)
        n = (x - self.mean.astype(np.float32)) / (
            np.sqrt(self.var.astype(np.float32)) + self.eps)
        return np.clip(n, -self.clip, self.clip).astype(np.float32)


class _ACPLReplayBuffer:
    def __init__(self, capacity=8000, seed=42):
        self.cap  = capacity
        self.rng  = np.random.default_rng(seed)
        self._buf = []
        self._pos = 0

    def push(self, state, action, reward, next_state, consequence, done, h, nh):
        entry = (np.array(state, np.float32), int(action), float(reward),
                 np.array(next_state, np.float32), float(consequence),
                 bool(done), np.array(h, np.float32), np.array(nh, np.float32),
                 1.0)
        if len(self._buf) < self.cap:
            self._buf.append(entry)
        else:
            self._buf[self._pos] = entry
        self._pos = (self._pos + 1) % self.cap

    def sample(self, batch_size):
        if len(self._buf) < batch_size:
            return None
        priors = np.array([e[8] for e in self._buf], np.float32)
        probs  = priors / (priors.sum() + 1e-9)
        idxs   = self.rng.choice(len(self._buf), batch_size, replace=False, p=probs)
        batch  = [self._buf[i] for i in idxs]
        S   = np.stack([b[0] for b in batch])
        A   = np.array([b[1] for b in batch], np.int32)
        R   = np.array([b[2] for b in batch], np.float32)
        NS  = np.stack([b[3] for b in batch])
        C   = np.array([b[4] for b in batch], np.float32)
        D   = np.array([b[5] for b in batch], np.float32)
        H   = np.stack([b[6] for b in batch])
        NH  = np.stack([b[7] for b in batch])
        W   = np.ones(batch_size, np.float32)
        return S, A, R, NS, C, D, H, NH, W, idxs

    def update_priorities(self, idxs, errors):
        for i, idx in enumerate(idxs):
            if idx < len(self._buf):
                e = self._buf[idx]
                self._buf[idx] = e[:8] + (abs(float(errors[i])) + 1e-6,)

    def __len__(self): return len(self._buf)


class _ACPLDuelingHead:
    def __init__(self, in_dim, n_actions, hidden_dim, rng, lr=8e-4):
        self.shared = _ACPLLinear(in_dim, hidden_dim, rng)
        self.val_h  = _ACPLLinear(hidden_dim, hidden_dim // 2, rng)
        self.val_o  = _ACPLLinear(hidden_dim // 2, 1, rng)
        self.adv_h  = _ACPLLinear(hidden_dim, hidden_dim // 2, rng)
        self.adv_o  = _ACPLLinear(hidden_dim // 2, n_actions, rng)
        all_p = (self.shared.params() + self.val_h.params() + self.val_o.params() +
                 self.adv_h.params() + self.adv_o.params())
        self.opt   = _ACPLAdam(all_p, lr=lr)
        self._cache = {}

    def forward(self, x):
        x  = np.asarray(x, np.float32)
        sh = _acpl_relu(self.shared.forward(x))
        vh = _acpl_relu(self.val_h.forward(sh))
        v  = self.val_o.forward(vh)
        ah = _acpl_relu(self.adv_h.forward(sh))
        a  = self.adv_o.forward(ah)
        q  = v + a - a.mean(axis=-1, keepdims=True)
        self._cache = {"x": x, "sh": sh, "vh": vh, "v": v, "ah": ah, "a": a}
        return q

    def backward_update(self, d_q):
        c = self._cache
        d_a  = d_q - d_q.mean(axis=-1, keepdims=True)
        d_ah_back, dW_ao, db_ao = self.adv_o.backward(d_a)
        d_ah2 = d_ah_back * _acpl_d_relu(c["ah"])
        d_sh_adv, dW_ah, db_ah = self.adv_h.backward(d_ah2)
        d_v  = d_q.sum(axis=-1, keepdims=True)
        d_vh_back, dW_vo, db_vo = self.val_o.backward(d_v)
        d_vh2 = d_vh_back * _acpl_d_relu(c["vh"])
        d_sh_val, dW_vh, db_vh = self.val_h.backward(d_vh2)
        d_sh  = (d_sh_adv + d_sh_val) * _acpl_d_relu(c["sh"])
        _, dW_s, db_s = self.shared.backward(d_sh)
        self.opt.step([dW_s, db_s, dW_vh, db_vh, dW_vo, db_vo,
                       dW_ah, db_ah, dW_ao, db_ao])

    def params_flat(self):
        return (self.shared.params() + self.val_h.params() + self.val_o.params() +
                self.adv_h.params() + self.adv_o.params())


class _ACPLNStepBuffer:
    def __init__(self, n: int = 3, gamma: float = 0.97):
        self.n     = n
        self.gamma = gamma
        self._buf: deque = deque(maxlen=n)

    def push(self, state, action, reward, next_state, consequence, done, h, nh):
        self._buf.append((state, action, float(reward), next_state,
                          float(consequence), bool(done), h, nh))

    def ready(self) -> bool:
        return len(self._buf) >= self.n

    def get(self) -> Optional[tuple]:
        if not self.ready():
            return None
        G    = 0.0
        C    = 0.0
        gk   = 1.0
        done = False
        last = self._buf[-1]           # default: last element in window
        for exp in self._buf:
            G    += gk * exp[2]
            C     = max(C, exp[4])
            gk   *= self.gamma
            if exp[5]:                 # terminal mid-window: use this element's next_state
                last = exp
                done = True
                break
        first = self._buf[0]
        return (first[0], first[1], G, last[3], C, done, first[6], last[7])


class ACPLAdvisor:
    """
    ACPL v2 — Adaptive Consequence-Penalised Learning.
    Original algorithm by Shashank Dev.

    State vector (14-dim):
      [0]  patient acuity
      [1]  wait pressure
      [2]  deterioration velocity
      [3]  resource utilisation
      [4]  skill match
      [5]  occupancy pressure
      [6]  mortality pressure
      [7]  predicted outcome probability
      [8]  patient category (normalised)
      [9]  resource remaining capacity fraction
      [10] time-of-day sin
      [11] time-of-day cos
      [12] demand pressure
      [13] resource fatigue
    """
    ACPL_STATE_DIM   = 14
    ACPL_GRU_DIM     = 32
    ACPL_HIDDEN_DIM  = 64
    ACPL_CONS_DIM    = 32
    ACPL_LAM_DIM     = 16
    ACPL_BATCH_SIZE  = 8
    ACPL_UPDATE_FREQ = 5
    ACPL_COST_SCALE  = 12.0
    ACPL_MAX_PEN     = 40.0
    ACPL_SIGMA_W     = 0.20
    ACPL_LAMBDA_MAX  = 2.0
    ACPL_GAMMA       = 0.97
    ACPL_CONSEQUENCE_DELAY = 5
    ACPL_NSTEP       = 3

    def __init__(self, seed: int = 2025) -> None:
        import threading, queue as _queue
        rng = np.random.default_rng(seed)
        D   = self.ACPL_STATE_DIM

        self._gru      = _ACPLGRU(D, self.ACPL_GRU_DIM, rng)
        self._q_head   = _ACPLDuelingHead(self.ACPL_GRU_DIM, 2,
                                          self.ACPL_HIDDEN_DIM, rng, lr=8e-4)
        self._q_params = self._gru.params() + self._q_head.params_flat()
        self._q_opt    = _ACPLAdam(self._q_params, lr=8e-4)

        self._tgt_gru  = _ACPLGRU(D, self.ACPL_GRU_DIM, rng)
        self._tgt_q    = _ACPLDuelingHead(self.ACPL_GRU_DIM, 2,
                                          self.ACPL_HIDDEN_DIM, rng, lr=8e-4)
        self._sync_target()

        self._cons_net  = _ACPLMiniNet(D, 2, self.ACPL_CONS_DIM, rng, lr=4e-4)
        self._lam_net   = _ACPLMiniNet(D, 1, self.ACPL_LAM_DIM,  rng, lr=2e-4)

        self._norm      = _ACPLStateNorm(D)
        self._buf       = _ACPLReplayBuffer(capacity=8000, seed=seed)
        self._nstep     = _ACPLNStepBuffer(n=self.ACPL_NSTEP, gamma=self.ACPL_GAMMA)
        self._h         = self._gru.zero_state(1)
        self._pending_cons: deque = deque(maxlen=200)

        self.last_policy_loss = 0.0
        self.last_cons_loss   = 0.0
        self.last_mean_lambda = 1.0
        self.last_mean_C      = 0.0
        self._tick            = 0
        self._update_count    = 0

        self._learn_queue = _queue.Queue(maxsize=400)
        self._learn_thread = threading.Thread(
            target=self._learn_loop, name="ACPL-Learn", daemon=True)
        self._learn_thread.start()
        LOG.info("ACPLAdvisor v2 — state_dim=%d  gru=%d  head=Dueling  n-step=%d",
                 D, self.ACPL_GRU_DIM, self.ACPL_NSTEP)

    def _learn_loop(self):
        while True:
            try:
                item = self._learn_queue.get(timeout=2.0)
            except Exception:
                continue
            if item is None:
                break
            try:
                state, action, reward, next_state, consequence, done, h, nh = item
                self._nstep.push(state, action, reward, next_state,
                                 consequence, done, h, nh)
                if self._nstep.ready():
                    agg = self._nstep.get()
                    if agg is not None:
                        self._buf.push(*agg)
                        self.maybe_update()
            except Exception as exc:
                LOG.debug("ACPL learn_loop error: %s", exc)

    def stop(self):
        try:
            self._learn_queue.put_nowait(None)
        except Exception:
            pass

    @staticmethod
    def build_state(acuity: float, wait_pressure: float, deterioration_vel: float,
                    resource_load: float, skill_match: float,
                    occupancy: float, mortality_pressure: float,
                    pred_outcome: float, patient_cat_int: int,
                    resource_cap_frac: float,
                    time_of_day: float = 0.0,
                    demand_pressure: float = 0.5,
                    resource_fatigue: float = 0.0) -> np.ndarray:
        tod_angle = 2.0 * math.pi * time_of_day / 24.0
        return np.array([
            float(np.clip(acuity,              0.0, 1.0)),
            float(np.clip(wait_pressure,       0.0, 1.0)),
            float(np.clip(deterioration_vel,   0.0, 1.0)),
            float(np.clip(resource_load,       0.0, 1.0)),
            float(np.clip(skill_match,         0.0, 1.0)),
            float(np.clip(occupancy / 1.2,     0.0, 1.0)),
            float(np.clip(mortality_pressure,  0.0, 1.0)),
            float(np.clip(pred_outcome,        0.0, 1.0)),
            float(np.clip(patient_cat_int / 5.0, 0.0, 1.0)),
            float(np.clip(resource_cap_frac,   0.0, 1.0)),
            float((math.sin(tod_angle) + 1.0) * 0.5),
            float((math.cos(tod_angle) + 1.0) * 0.5),
            float(np.clip(demand_pressure,     0.0, 1.0)),
            float(np.clip(resource_fatigue,    0.0, 1.0)),
        ], np.float32)

    def penalty(self, state: np.ndarray) -> float:
        s    = self._norm.normalize(state)[None]
        self._gru.forward(s, self._h)
        cons_out = self._cons_net.forward(s)
        C_pred   = float(np.clip(_acpl_sigmoid(cons_out[0, 0]), 0.0, 1.0))
        log_sigma= float(cons_out[0, 1])
        sigma    = float(np.clip(np.exp(np.clip(log_sigma, -5, 2)), 0.01, 1.0))
        lam_out  = self._lam_net.forward(s)
        lam      = float(np.clip(_acpl_sigmoid(lam_out[0, 0]) * self.ACPL_LAMBDA_MAX,
                                 0.0, self.ACPL_LAMBDA_MAX))
        pen = lam * (C_pred + self.ACPL_SIGMA_W * sigma) * self.ACPL_COST_SCALE
        pen = max(pen, self.ACPL_MAX_PEN * 0.20 * C_pred)
        return float(np.clip(pen, 0.0, self.ACPL_MAX_PEN))

    def advance_hidden(self, state: np.ndarray):
        s = self._norm.normalize(state)[None]
        self._h = self._gru.forward(s, self._h)

    def reset_hidden(self):
        self._h = self._gru.zero_state(1)

    def feedback(self, state, action, reward, next_state, done, consequence=0.0):
        self._norm.update(state)
        h  = self._h.copy().squeeze()
        nh = self._gru.forward(
            self._norm.normalize(next_state)[None], self._h
        ).squeeze()
        experience = (state, action, reward, next_state, consequence, done, h, nh)
        try:
            self._learn_queue.put_nowait(experience)
        except Exception:
            pass

    def queue_pending_consequence(self, tick_now, state, action, reward, next_state):
        self._pending_cons.append((tick_now + self.ACPL_CONSEQUENCE_DELAY,
                                   state, action, reward, next_state))

    def flush_pending(self, tick_now, consequence_signal=0.0):
        while self._pending_cons:
            due_tick, s, a, r, ns = self._pending_cons[0]
            if tick_now < due_tick:
                break
            self._pending_cons.popleft()
            self.feedback(s, a, r, ns, done=True, consequence=consequence_signal)

    def maybe_update(self):
        self._tick += 1
        if self._tick % self.ACPL_UPDATE_FREQ != 0:
            return False
        batch = self._buf.sample(self.ACPL_BATCH_SIZE)
        if batch is None:
            return False
        S, A, R, NS, C, D, H, NH, W, idxs = batch
        B = len(S)
        S_norm  = self._norm.normalize(S)
        NS_norm = self._norm.normalize(NS)

        cons_out = self._cons_net.forward(S_norm)
        C_pred   = _acpl_sigmoid(cons_out[:, 0])
        log_sig  = cons_out[:, 1]
        c_err    = C_pred - C
        log_sig_err = log_sig - np.log(np.abs(c_err) + 1e-6)
        d_cons   = np.zeros_like(cons_out)
        d_cons[:, 0] = c_err * _acpl_d_sigmoid(C_pred) / B
        d_cons[:, 1] = np.clip(log_sig_err, -1.0, 1.0) / B
        self._cons_net.backward_update(d_cons)
        c_loss   = float(np.mean(c_err**2))

        h_pol_ns = self._gru.forward(NS_norm, NH)
        h_tgt_ns = self._tgt_gru.forward(NS_norm, NH)
        q_pol_ns = self._q_head.forward(h_pol_ns)
        q_tgt_ns = self._tgt_q.forward(h_tgt_ns)
        next_acts = q_pol_ns.argmax(axis=-1)
        next_q    = q_tgt_ns[np.arange(B), next_acts]

        cons_pen  = np.clip(C_pred * self.ACPL_LAMBDA_MAX, 0.0, np.abs(R) * 0.5 + 0.5)
        corrected_R = R - cons_pen
        td_target = corrected_R + (self.ACPL_GAMMA ** self.ACPL_NSTEP) * next_q * (1.0 - D)

        h_cur = self._gru.forward(S_norm, H)
        q_cur_all = self._q_head.forward(h_cur)
        q_cur = q_cur_all[np.arange(B), np.clip(A, 0, 1)]
        td_err = td_target - q_cur
        p_loss = float(np.mean(W * td_err**2))

        d_q = np.zeros_like(q_cur_all)
        d_q[np.arange(B), np.clip(A, 0, 1)] = -2.0 * W * td_err / B
        self._q_head.backward_update(d_q)

        lam_out  = self._lam_net.forward(S_norm)
        lam_pred = _acpl_sigmoid(lam_out[:, 0]) * self.ACPL_LAMBDA_MAX
        c_norm   = C / (np.abs(C).mean() + 1e-6)
        lam_tgt  = _acpl_sigmoid(c_norm - 1.0)
        lam_err  = (lam_pred / self.ACPL_LAMBDA_MAX) - lam_tgt
        d_lam    = np.zeros_like(lam_out)
        d_lam[:, 0] = lam_err * _acpl_d_sigmoid(lam_pred / self.ACPL_LAMBDA_MAX) / B
        self._lam_net.backward_update(d_lam)

        for tp, pp in zip(self._tgt_gru.params(), self._gru.params()):
            tp[:] = 0.01 * pp + 0.99 * tp
        for tp, pp in zip(self._tgt_q.params_flat(), self._q_head.params_flat()):
            tp[:] = 0.01 * pp + 0.99 * tp

        self._buf.update_priorities(idxs, np.abs(td_err))
        self._update_count    += 1
        self.last_policy_loss  = p_loss
        self.last_cons_loss    = c_loss
        self.last_mean_lambda  = float(lam_pred.mean())
        self.last_mean_C       = float(C_pred.mean())
        return True

    def _sync_target(self):
        for tp, pp in zip(self._tgt_gru.params(), self._gru.params()):
            tp[:] = pp
        for tp, pp in zip(self._tgt_q.params_flat(), self._q_head.params_flat()):
            tp[:] = pp

    def diagnostics(self) -> dict:
        return {
            "acpl_updates":  self._update_count,
            "acpl_buf_size": len(self._buf),
            "acpl_p_loss":   round(self.last_policy_loss, 4),
            "acpl_c_loss":   round(self.last_cons_loss,   4),
            "acpl_mean_lam": round(self.last_mean_lambda, 4),
            "acpl_mean_C":   round(self.last_mean_C,      4),
        }


import matplotlib
_BACKENDS = ["TkAgg", "Qt5Agg", "Qt6Agg", "WxAgg", "MacOSX", "Agg"]
for _b in _BACKENDS:
    try:
        matplotlib.use(_b)
        import matplotlib.pyplot as _chk
        _chk.figure(); _chk.close("all")
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

_SESSION_ID = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

def _build_logger(name="HRMS", level=logging.INFO, output_dir=".") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s  [%(levelname)-5s]  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level); ch.setFormatter(fmt)
    logger.addHandler(ch)
    try:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"hrms_{_SESSION_ID}.log"
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger._log_file_path = str(log_path)
    except OSError as e:
        logger.warning("Could not create log file: %s", e)
    return logger

LOG = _build_logger()

def _reinit_file_logger(output_dir: str) -> None:
    logger = logging.getLogger("HRMS")
    fmt = logging.Formatter("%(asctime)s  [%(levelname)-5s]  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    for h in list(logger.handlers):
        if isinstance(h, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
            h.close(); logger.removeHandler(h)
    try:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"hrms_{_SESSION_ID}.log"
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger._log_file_path = str(log_path)
        LOG.info("Log file: %s", log_path)
    except OSError as e:
        LOG.warning("Could not create log file: %s", e)


class DemandForecaster:
    def __init__(self, alpha: float = 0.30, baseline: float = 3.5):
        self.alpha   = alpha
        self._ema    = baseline
        self._var    = 2.0
        self._sizes: deque = deque(maxlen=20)
        self.surge_level  = 0.0

    def record_wave(self, n_patients: int) -> None:
        self._sizes.append(n_patients)
        old_ema = self._ema
        self._ema = self.alpha * n_patients + (1 - self.alpha) * self._ema
        self._var = (self.alpha * (n_patients - old_ema) ** 2
                     + (1 - self.alpha) * self._var)
        if len(self._sizes) >= 3:
            recent = float(np.mean(list(self._sizes)[-3:]))
            self.surge_level = float(np.clip((recent - 3.5) / 4.5, 0.0, 1.0))

    def predict_next(self) -> Tuple[float, float]:
        return self._ema, float(np.sqrt(max(self._var, 0.5)))

    def predict_pressure(self) -> float:
        return float(np.clip(self._ema / 8.0 + self.surge_level * 0.4, 0.0, 1.0))

    def is_surge(self) -> bool:
        return self.surge_level > 0.5

    def forecast_str(self) -> str:
        mu, sigma = self.predict_next()
        flag = " ⚠ SURGE" if self.is_surge() else ""
        return f"Next wave ≈{mu:.1f}±{sigma:.1f} pts{flag}"


class SLATracker:
    SLA_LIMITS = {
        0: 150,   # EMERGENCY
        1: 100,   # ICU
        2: 600,   # GENERAL
        3: 450,   # SURGICAL
        4: 300,   # MATERNITY
    }

    def __init__(self):
        self._violations: Dict[int, int] = defaultdict(int)
        self._compliant:  Dict[int, int] = defaultdict(int)
        self._breach_log: deque = deque(maxlen=50)

    def check(self, patient: "Patient", sim_t: float) -> bool:
        limit = self.SLA_LIMITS.get(int(patient.category), 600)
        cat   = int(patient.category)
        if patient.assigned_resource is None and patient.ticks_waiting > limit:
            self._violations[cat] += 1
            self._breach_log.append(
                f"[SLA BREACH T+{sim_t:.0f}h] P{patient.uid:03d} "
                f"{patient.name_str()} waited {patient.ticks_waiting} ticks")
            return True
        self._compliant[cat] += 1
        return False

    def violation_rate(self) -> float:
        total_v = sum(self._violations.values())
        total_c = sum(self._compliant.values())
        return total_v / max(1, total_v + total_c)

    def per_category_rates(self) -> Dict[str, float]:
        from_int = {0:"EMERGENCY",1:"ICU",2:"GENERAL",3:"SURGICAL",4:"MATERNITY"}
        out = {}
        for k, name in from_int.items():
            v = self._violations.get(k, 0)
            c = self._compliant.get(k, 0)
            out[name] = round(v / max(1, v + c), 3)
        return out

    def total_violations(self) -> int:
        return sum(self._violations.values())

    def latest_breaches(self, n: int = 3) -> List[str]:
        return list(self._breach_log)[-n:]


class ReadmissionQueue:
    READMISSION_PROB  = 0.05
    READMISSION_DELAY = (50, 200)

    def __init__(self, seed: int = 2025):
        self._rng: np.random.Generator = np.random.default_rng(seed + 42)
        self._queue: List[Tuple[int, int, float]] = []

    def schedule(self, patient: "Patient", tick: int) -> bool:
        if self._rng.random() < self.READMISSION_PROB:
            delay  = int(self._rng.integers(*self.READMISSION_DELAY))
            ra = float(np.clip(patient.acuity + self._rng.uniform(0.15, 0.35), 0.4, 0.95))
            self._queue.append((tick + delay, int(patient.category), ra))
            return True
        return False

    def flush(self, tick: int) -> List[Tuple[int, float]]:
        due   = [(c, a) for (t, c, a) in self._queue if t <= tick]
        self._queue = [(t, c, a) for (t, c, a) in self._queue if t > tick]
        return due

    def pending(self) -> int:
        return len(self._queue)


class SeedManager:
    _master_seed: int = 2025
    _streams_cls: dict = {}

    def __init__(self, master_seed=2025):
        self._master_seed = master_seed
        self._inst_streams: Dict[str, np.random.Generator] = {}

    def _inst_get(self, name: str) -> np.random.Generator:
        if name not in self._inst_streams:
            sub = (self._master_seed + hash(name)) & 0xFFFFFFFF
            self._inst_streams[name] = np.random.default_rng(sub)
        return self._inst_streams[name]

    @classmethod
    def set_master(cls, seed: int):
        cls._master_seed = seed
        cls._streams_cls = {}

    @classmethod
    def get_cls(cls, name: str) -> np.random.Generator:
        if name not in cls._streams_cls:
            sub = (cls._master_seed + hash(name)) & 0xFFFFFFFF
            cls._streams_cls[name] = np.random.default_rng(sub)
        return cls._streams_cls[name]

class _GetDescriptor:
    def __get__(self, obj, objtype=None):
        if obj is None:
            return SeedManager.get_cls
        return obj._inst_get

SeedManager.get = _GetDescriptor()


@dataclass
class ExperimentConfig:
    seed:              int   = 2025
    num_resources:     int   = 10
    max_patients:      int   = 16
    admission_period:  int   = 90
    max_frames:        int   = 2000
    anim_interval:     int   = 45
    dt:                float = 0.08
    export_csv:        bool  = True
    log_level:         int   = logging.INFO
    experiment_id:     str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    output_dir:        str   = "hrms_runs"
    fig_width:         float = 26.0
    fig_height:        float = 14.0
    theme:             str   = "dark"
    batch_runs:        int   = 1
    ehr_loss_prob:     float = 0.04
    ehr_delay_t:       int   = 1
    assign_delay_t:    int   = 1
    assign_dwell_min:  int   = 12
    assign_switch_cost:float = 40.0
    enable_fatigue:    bool  = True
    enable_readmission:bool  = True
    night_shift:       bool  = False
    scenario_label:    str   = ""

    def __post_init__(self):
        if not (1 <= self.num_resources <= 40):
            raise ValueError(f"num_resources must be 1-40, got {self.num_resources}")
        if not (1 <= self.max_patients <= 60):
            raise ValueError(f"max_patients must be 1-60, got {self.max_patients}")
        if self.theme not in ("dark", "light"):
            raise ValueError(f"theme must be dark|light, got {self.theme}")


_PAL_DARK = {
    "bg":     "#0a0f14", "panel":  "#111820", "border": "#1e2c3a",
    "grid":   "#162030", "white":  "#e8edf2", "dim":    "#5a7a8a",
    "green":  "#00e5a0", "red":    "#ff4040", "yellow": "#ffd060",
    "cyan":   "#40d0ff", "orange": "#ff8040", "purple": "#a060ff",
}
_PAL_LIGHT = {
    "bg":     "#f0f4f8", "panel":  "#ffffff", "border": "#d0dce8",
    "grid":   "#e8eef4", "white":  "#1a2530", "dim":    "#607080",
    "green":  "#009060", "red":    "#cc2020", "yellow": "#b07000",
    "cyan":   "#0080b0", "orange": "#c05010", "purple": "#6030c0",
}
PAL = _PAL_DARK


class PatientCategory(IntEnum):
    EMERGENCY = 0
    ICU       = 1
    GENERAL   = 2
    SURGICAL  = 3
    MATERNITY = 4

_DISCHARGE_ACUITY = {
    PatientCategory.EMERGENCY: 0.25,
    PatientCategory.ICU:       0.20,
    PatientCategory.GENERAL:   0.30,
    PatientCategory.SURGICAL:  0.28,
    PatientCategory.MATERNITY: 0.22,
}

_RES_DEMAND = {
    PatientCategory.EMERGENCY: 2,
    PatientCategory.ICU:       3,
    PatientCategory.GENERAL:   1,
    PatientCategory.SURGICAL:  2,
    PatientCategory.MATERNITY: 1,
}

_CAT_NAMES = {
    PatientCategory.EMERGENCY: "EMERGENCY",
    PatientCategory.ICU:       "ICU",
    PatientCategory.GENERAL:   "GENERAL",
    PatientCategory.SURGICAL:  "SURGICAL",
    PatientCategory.MATERNITY: "MATERNITY",
}

CRITICAL_ACUITY  = 0.85
MORTALITY_ACUITY = 1.0

_CAT_COLORS = {
    PatientCategory.EMERGENCY: "#ff4040",
    PatientCategory.ICU:       "#ff8040",
    PatientCategory.GENERAL:   "#40d0ff",
    PatientCategory.SURGICAL:  "#a060ff",
    PatientCategory.MATERNITY: "#00e5a0",
}


class ResourceStatus(IntEnum):
    IDLE        = 0
    ASSIGNED    = 1
    OVERLOADED  = 2
    MAINTENANCE = 3

class ResourceType(IntEnum):
    GENERAL_BED    = 0
    ICU_BED        = 1
    EMERGENCY_BAY  = 2
    SURGICAL_SUITE = 3
    STAFF_TEAM     = 4

_RES_CAPACITY = {
    ResourceType.GENERAL_BED:    4,
    ResourceType.ICU_BED:        2,
    ResourceType.EMERGENCY_BAY:  3,
    ResourceType.SURGICAL_SUITE: 2,
    ResourceType.STAFF_TEAM:     5,
}

_RES_NAMES = {
    ResourceType.GENERAL_BED:    "GEN-BED",
    ResourceType.ICU_BED:        "ICU-BED",
    ResourceType.EMERGENCY_BAY:  "EMRG-BAY",
    ResourceType.SURGICAL_SUITE: "SURG",
    ResourceType.STAFF_TEAM:     "STAFF",
}

_SKILL_MATCH = {
    ResourceType.GENERAL_BED:    {0:0.4, 1:0.2, 2:1.0, 3:0.5, 4:0.7},
    ResourceType.ICU_BED:        {0:0.6, 1:1.0, 2:0.3, 3:0.5, 4:0.3},
    ResourceType.EMERGENCY_BAY:  {0:1.0, 1:0.6, 2:0.4, 3:0.5, 4:0.5},
    ResourceType.SURGICAL_SUITE: {0:0.3, 1:0.3, 2:0.3, 3:1.0, 4:0.3},
    ResourceType.STAFF_TEAM:     {0:0.7, 1:0.7, 2:0.7, 3:0.7, 4:0.8},
}


_pid_counter = 0

class Patient:
    def __init__(self, cfg: ExperimentConfig, category: PatientCategory,
                 wave: int, rng: np.random.Generator,
                 initial_acuity: Optional[float] = None,
                 readmission: bool = False):
        global _pid_counter
        _pid_counter += 1
        self.uid       = _pid_counter
        self.category  = category
        self.wave      = wave
        self.active    = True
        self.cfg       = cfg
        self.readmission = readmission

        base = {
            PatientCategory.EMERGENCY: 0.55,
            PatientCategory.ICU:       0.70,
            PatientCategory.GENERAL:   0.35,
            PatientCategory.SURGICAL:  0.45,
            PatientCategory.MATERNITY: 0.30,
        }[category]
        if initial_acuity is not None:
            self.acuity = float(np.clip(initial_acuity, 0.15, 0.95))
        else:
            self.acuity = float(np.clip(base + rng.normal(0, 0.10), 0.15, 0.95))
        self.acuity_hist: deque = deque(maxlen=50)

        self.comorbidity_risk = float(rng.beta(2, 5))

        base_det = {
            PatientCategory.EMERGENCY: 0.008, PatientCategory.ICU: 0.012,
            PatientCategory.GENERAL:   0.003, PatientCategory.SURGICAL: 0.005,
            PatientCategory.MATERNITY: 0.002,
        }[category]
        self.det_rate = float(np.clip(
            base_det * (1.0 + self.comorbidity_risk) + rng.normal(0, 0.001),
            0.001, 0.025))

        base_imp = {
            PatientCategory.EMERGENCY: 0.015, PatientCategory.ICU: 0.010,
            PatientCategory.GENERAL:   0.020, PatientCategory.SURGICAL: 0.018,
            PatientCategory.MATERNITY: 0.025,
        }[category]
        self.imp_rate = float(np.clip(base_imp + rng.normal(0, 0.002), 0.005, 0.040))

        self.assigned_resource: Optional["Resource"] = None
        self.ticks_waiting  = 0
        self.ticks_in_care  = 0
        self.age            = 0
        self.treatment_duration = int(rng.integers(60, 300))
        self.danger_level   = 0
        self._sla_breached  = False

    def step(self, dt: float, rng: np.random.Generator,
             resource_effectiveness: float = 1.0) -> None:
        if not self.active:
            return
        self.age += 1
        self.acuity_hist.append(self.acuity)

        if self.assigned_resource is not None:
            sm = _SKILL_MATCH[self.assigned_resource.rtype][int(self.category)]
            effective_imp = self.imp_rate * (0.5 + 0.5 * sm) * resource_effectiveness
            self.acuity = max(0.0, self.acuity - effective_imp * dt * 60)
            self.ticks_in_care += 1
        else:
            self.acuity = min(MORTALITY_ACUITY, self.acuity + self.det_rate * dt * 60)
            self.ticks_waiting += 1

        noise = rng.normal(0, 0.003)
        self.acuity = float(np.clip(self.acuity + noise, 0.0, MORTALITY_ACUITY))

        if   self.acuity > 0.85: self.danger_level = 3
        elif self.acuity > 0.65: self.danger_level = 2
        elif self.acuity > 0.45: self.danger_level = 1
        else:                    self.danger_level = 0

    def deterioration_velocity(self) -> float:
        if len(self.acuity_hist) < 2:
            return self.det_rate
        hist = list(self.acuity_hist)[-10:]
        if len(hist) < 2:
            return self.det_rate
        return max(0.0, (hist[-1] - hist[0]) / max(1, len(hist) - 1))

    def priority_score(self) -> float:
        wait_mod = 1.0 + min(self.ticks_waiting / 100.0, 2.0)
        return float(self.acuity * wait_mod)

    def name_str(self) -> str:
        return _CAT_NAMES[self.category]

    def is_dischargeable(self) -> bool:
        return (self.acuity <= _DISCHARGE_ACUITY[self.category]
                and self.ticks_in_care >= self.treatment_duration * 0.5)

    def __repr__(self):
        return f"Patient(uid={self.uid}, cat={self.name_str()}, acuity={self.acuity:.2f})"


class Resource:
    _FATIGUE_RATE    = 0.0015
    _RECOVERY_RATE   = 0.0006

    def __init__(self, idx: int, rtype: ResourceType, cfg: ExperimentConfig,
                 rng: np.random.Generator):
        self.idx      = idx
        self.rtype    = rtype
        self.cfg      = cfg
        self.status   = ResourceStatus.IDLE
        self.capacity = _RES_CAPACITY[rtype]
        self.used     = 0
        self.patient: Optional[Patient] = None
        self.kills_total = 0
        self.overloads   = 0
        self._dwell_ticks      = 0
        self._last_patient_uid = -1
        self.maintenance_t     = 0
        self.util_hist: deque  = deque(maxlen=200)
        self.load = float(rng.uniform(0.0, 0.3))
        self.fatigue: float = 0.0
        self.effectiveness: float = 1.0
        self._night_capacity_override: Optional[int] = None

    @property
    def available_slots(self) -> int:
        cap = self._night_capacity_override if self._night_capacity_override is not None \
              else self.capacity
        return max(0, cap - self.used)

    def can_accept(self, pat: Patient) -> bool:
        demand = _RES_DEMAND[pat.category]
        return (self.status != ResourceStatus.MAINTENANCE
                and self.available_slots >= demand)

    def assign(self, pat: Patient) -> bool:
        demand = _RES_DEMAND[pat.category]
        if not self.can_accept(pat):
            return False
        self.used += demand
        self.patient = pat
        pat.assigned_resource = self
        self.status = ResourceStatus.ASSIGNED if self.used < self.capacity else ResourceStatus.OVERLOADED
        return True

    def release(self, pat: Patient) -> None:
        demand = _RES_DEMAND[pat.category]
        self.used = max(0, self.used - demand)
        if pat.assigned_resource is self:
            pat.assigned_resource = None
        self.patient = None
        self.status = ResourceStatus.IDLE if self.used == 0 else ResourceStatus.ASSIGNED

    def step(self, dt: float, enable_fatigue: bool = True) -> None:
        if self.maintenance_t > 0:
            self.maintenance_t -= 1
            if self.maintenance_t == 0:
                self.status = ResourceStatus.IDLE
                self.used   = 0

        target_load = self.used / max(1, self.capacity)
        self.load = 0.95 * self.load + 0.05 * target_load
        self.util_hist.append(self.load)

        if enable_fatigue and self.rtype == ResourceType.STAFF_TEAM:
            if self.load > 0.70:
                self.fatigue = min(1.0, self.fatigue + self._FATIGUE_RATE * self.load)
            elif self.load < 0.30:
                self.fatigue = max(0.0, self.fatigue - self._RECOVERY_RATE)
            self.effectiveness = max(0.60, 1.0 - 0.40 * self.fatigue)
        else:
            self.effectiveness = 1.0

        if self.patient and self.patient.active:
            if self.patient.uid == self._last_patient_uid:
                self._dwell_ticks += 1
            else:
                self._dwell_ticks = 0
                self._last_patient_uid = self.patient.uid

    def can_reassign(self) -> bool:
        return self._dwell_ticks >= self.cfg.assign_dwell_min

    def name_str(self) -> str:
        return f"{_RES_NAMES[self.rtype]}-{self.idx:02d}"

    def __repr__(self):
        return (f"Resource({self.name_str()}, load={self.load:.2f}, "
                f"fatigue={self.fatigue:.2f}, used={self.used}/{self.capacity})")


class AcuityFilter:
    def __init__(self, initial_acuity: float):
        self.x = np.array([initial_acuity, 0.0], np.float64)
        self.P = np.eye(2) * 0.1
        self.Q = np.diag([1e-4, 1e-5])
        self.R = 0.01
        self.F = np.array([[1, 1], [0, 0.98]], np.float64)
        self.H = np.array([[1, 0]], np.float64)
        self.confidence = 0.5
        self._meas_hist: deque = deque(maxlen=50)

    def predict(self, dt: float = 1.0) -> np.ndarray:
        F = np.array([[1, dt], [0, 0.98 ** dt]], np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, acuity_meas: float) -> None:
        self._meas_hist.append(acuity_meas)
        z = np.array([acuity_meas], np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / (S + 1e-9)
        self.x = self.x + K.flatten() * y[0]
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.confidence = min(1.0, self.confidence + 0.05)

    def future(self, steps: int) -> float:
        x = self.x.copy()
        for _ in range(steps):
            x = np.array([[1, 1], [0, 0.98]]) @ x
        return float(np.clip(x[0], 0.0, 1.0))

    def facuity(self) -> float:
        return float(np.clip(self.x[0], 0.0, 1.0))

    def fvel(self) -> float:
        return float(self.x[1])


@dataclass
class AllocationEvent:
    tick:             int
    sim_time:         float
    outcome:          str
    patient_uid:      int
    patient_category: str
    patient_wave:     int
    resource_idx:     int
    resource_type:    str
    acuity_at_event:  float
    wait_ticks:       int
    care_ticks:       int
    occupancy_at_evt: float
    acpl_penalty:     float
    readmission:      bool  = False


class MetricsStore:
    def __init__(self):
        self._events: List[AllocationEvent] = []
        self._pipeline_lat: deque = deque(maxlen=500)
        self._cat_discharge: Dict[str, int] = defaultdict(int)
        self._cat_mortality: Dict[str, int] = defaultdict(int)
        self._readmissions:  int = 0

    def push_event(self, evt: AllocationEvent):
        self._events.append(evt)
        if evt.outcome == "discharged":
            self._cat_discharge[evt.patient_category] += 1
            if evt.readmission:
                self._readmissions += 1
        elif evt.outcome == "mortality":
            self._cat_mortality[evt.patient_category] += 1

    def push_lat(self, ms: float):
        self._pipeline_lat.append(ms)

    def per_category_summary(self) -> Dict[str, dict]:
        cats = ["EMERGENCY","ICU","GENERAL","SURGICAL","MATERNITY"]
        return {
            c: {"discharged": self._cat_discharge.get(c, 0),
                "mortalities": self._cat_mortality.get(c, 0)}
            for c in cats
        }

    def export_csv(self, path: str):
        if not self._events:
            return
        fields = list(self._events[0].__dataclass_fields__.keys())
        try:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for e in self._events:
                    w.writerow({k: getattr(e, k) for k in fields})
        except OSError as e:
            LOG.warning("CSV export failed: %s", e)


class WardLayout:
    GRID = 64

    def __init__(self, rng: np.random.Generator):
        G = self.GRID
        grid = np.zeros((G, G), dtype=np.int32)
        grid[2:30, 2:62] = 1
        grid[2:18, 40:62] = 2
        grid[34:62, 2:26] = 3
        grid[34:62, 36:62] = 4
        grid[34:62, 26:36] = 5
        self.grid = grid
        self._build_image()
        self.heatmap = np.zeros((G, G), dtype=float)

    def _build_image(self):
        G = self.GRID
        zone_colors = np.array([
            [0.05, 0.10, 0.15], [0.05, 0.20, 0.10], [0.20, 0.05, 0.05],
            [0.20, 0.10, 0.02], [0.08, 0.08, 0.22], [0.15, 0.05, 0.20],
        ], np.float32)
        img = np.empty((G, G, 4), np.float32)
        img[..., :3] = zone_colors[self.grid]
        img[..., 3]  = 0.85
        self.image = img

    def zone_at(self, pat: "Patient") -> int:
        return {PatientCategory.GENERAL:1, PatientCategory.ICU:2,
                PatientCategory.EMERGENCY:3, PatientCategory.SURGICAL:4,
                PatientCategory.MATERNITY:5}.get(pat.category, 0)

    def patient_display_pos(self, pat: "Patient", idx: int) -> Tuple[float, float]:
        G = self.GRID
        zone_centres = {
            PatientCategory.GENERAL:(32,16), PatientCategory.ICU:(51,10),
            PatientCategory.EMERGENCY:(14,48), PatientCategory.SURGICAL:(49,48),
            PatientCategory.MATERNITY:(31,48),
        }
        cx, cy = zone_centres.get(pat.category, (32, 32))
        ox = (idx % 5) * 4 - 8
        oy = (idx // 5) * 4 - 4
        return float(np.clip(cx + ox, 2, G-2)), float(np.clip(cy + oy, 2, G-2))

    def accumulate_heat(self, x, y, value=1.0):
        gx = int(np.clip(x, 0, self.GRID - 1))
        gy = int(np.clip(y, 0, self.GRID - 1))
        self.heatmap[gy, gx] += value


class ResourceAllocator:
    URGENT_PRIORITY  = 0.65
    SATURATION_DENSE = 1.2

    def __init__(self, cfg: ExperimentConfig, acpl: ACPLAdvisor):
        self.cfg  = cfg
        self.acpl = acpl
        self._occupancy = 0.0
        self.risk_cache: Dict[int, float] = {}
        self._last_avail_count = 0
        self._acpl_tick_state: Optional[np.ndarray] = None
        self._saturation_index = 0.0
        self._mortality_pressure = 0.0
        self.decision_log: deque = deque(maxlen=200)
        self.demand_pressure: float = 0.5
        self.sim_time: float = 0.0

    def update_risk(self, patients: List[Patient], resources: List[Resource]) -> None:
        total_slots = sum(r.capacity for r in resources)
        used_slots  = sum(r.used for r in resources)
        self._occupancy = used_slots / max(1, total_slots)
        self._saturation_index = self._occupancy
        critical = [p for p in patients if p.active and p.acuity >= CRITICAL_ACUITY]
        self._mortality_pressure = len(critical) / max(1, len([p for p in patients if p.active]))
        for p in patients:
            if p.active:
                self.risk_cache[p.uid] = p.priority_score()

    def assign(self, patients: List[Patient], resources: List[Resource],
               filters: Dict[int, "AcuityFilter"], tick: int, sim_t: float) -> None:
        free_patients = [p for p in patients if p.active and p.assigned_resource is None]
        if not free_patients:
            return
        avail_res = [r for r in resources
                     if r.status != ResourceStatus.MAINTENANCE
                     and r.available_slots > 0]
        if not avail_res:
            return

        N = len(avail_res); M = len(free_patients)
        C = np.full((N, M), 9999.0, np.float64)
        time_of_day = (sim_t % 24.0)

        for i, res in enumerate(avail_res):
            for j, pat in enumerate(free_patients):
                if not res.can_accept(pat):
                    continue
                filt     = filters.get(pat.uid)
                priority = self.risk_cache.get(pat.uid, pat.priority_score())
                urgency_pen = -priority * 20.0 * (1.0 + 0.3 * min(self._saturation_index, 1.5))
                sm          = _SKILL_MATCH[res.rtype][int(pat.category)]
                skill_pen   = -(sm - 0.5) * 15.0
                wait_pen    = -min(pat.ticks_waiting / 50.0, 3.0) * 8.0
                det_vel     = pat.deterioration_velocity()
                vel_pen     = -det_vel * 100.0
                load_pen    = res.load * 20.0
                fatigue_pen = res.fatigue * 15.0 if res.rtype == ResourceType.STAFF_TEAM else 0.0
                pred_pen    = 0.0
                if filt:
                    pred_acuity = filt.future(steps=10)
                    pred_pen    = -pred_acuity * 12.0
                else:
                    pred_acuity = pat.acuity
                switch_pen = 0.0
                if res.patient and res.patient.uid != pat.uid:
                    switch_pen = self.cfg.assign_switch_cost

                acpl_state = ACPLAdvisor.build_state(
                    acuity            = pat.acuity,
                    wait_pressure     = min(pat.ticks_waiting / 100.0, 1.0),
                    deterioration_vel = min(det_vel * 50, 1.0),
                    resource_load     = res.load,
                    skill_match       = sm,
                    occupancy         = self._occupancy,
                    mortality_pressure= self._mortality_pressure,
                    pred_outcome      = 1.0 - pred_acuity,
                    patient_cat_int   = int(pat.category),
                    resource_cap_frac = res.available_slots / max(1, res.capacity),
                    time_of_day       = time_of_day,
                    demand_pressure   = self.demand_pressure,
                    resource_fatigue  = res.fatigue,
                )
                acpl_pen = self.acpl.penalty(acpl_state)
                self._acpl_tick_state = acpl_state

                total = (urgency_pen + skill_pen + wait_pen + vel_pen
                         + load_pen + fatigue_pen + pred_pen + switch_pen + acpl_pen)
                C[i, j] = total

        rows, cols = linear_sum_assignment(C)
        for r_idx, p_idx in zip(rows, cols):
            if p_idx >= M or C[r_idx, p_idx] >= 9000.0:
                continue
            res = avail_res[r_idx]
            pat = free_patients[p_idx]
            if pat.assigned_resource is not None:
                continue
            is_urgent = pat.acuity >= self.URGENT_PRIORITY
            delay = 0 if is_urgent else self.cfg.assign_delay_t
            if delay == 0:
                res.assign(pat)

        self._last_avail_count = len(avail_res)

    def reassign_on_discharge(self, resources, patients, filters, tick, sim_t):
        self.assign(patients, resources, filters, tick, sim_t)


class Simulation:
    def __init__(self, cfg: ExperimentConfig):
        global PAL
        PAL = _PAL_LIGHT if cfg.theme == "light" else _PAL_DARK

        self.cfg  = cfg
        self.rng  = SeedManager(cfg.seed)
        global _pid_counter
        _pid_counter = 0

        self.acpl  = ACPLAdvisor(seed=cfg.seed)
        self.alloc = ResourceAllocator(cfg, self.acpl)
        self.ward  = WardLayout(self.rng.get("ward"))

        self.forecaster   = DemandForecaster()
        self.sla          = SLATracker()
        self.readmissions = ReadmissionQueue(seed=cfg.seed) if cfg.enable_readmission else None

        self.resources: List[Resource]          = self._init_resources()
        self.patients:  List[Patient]           = []
        self.filters:   Dict[int, AcuityFilter] = {}
        self.metrics    = MetricsStore()

        self.discharges    = 0
        self.unserved      = 0
        self.mortalities   = 0
        self.readmit_count = 0
        self.wave          = 0
        self.step_n        = 0
        self.t             = 0.0

        self.occupancy_hist:      deque = deque(maxlen=500)
        self.discharge_rate_hist: deque = deque(maxlen=500)
        self.patient_count_hist:  deque = deque(maxlen=500)
        self.mortality_hist:      deque = deque(maxlen=500)
        self.wait_time_hist:      deque = deque(maxlen=500)
        self.sla_violation_hist:  deque = deque(maxlen=500)
        self.fatigue_hist:        deque = deque(maxlen=500)
        self.events:              deque = deque(maxlen=30)
        self._timeline_events:    list  = []

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._csv_path      = str(Path(cfg.output_dir) / f"hrms_events_{cfg.experiment_id}.csv")
        self._last_csv_tick = 0

        self._spawn_wave()

    def _init_resources(self) -> List[Resource]:
        rng = self.rng.get("resources")
        n   = self.cfg.num_resources
        types = (
            [ResourceType.GENERAL_BED]    * max(1, n // 4) +
            [ResourceType.ICU_BED]        * max(1, n // 6) +
            [ResourceType.EMERGENCY_BAY]  * max(1, n // 5) +
            [ResourceType.SURGICAL_SUITE] * max(1, n // 6) +
            [ResourceType.STAFF_TEAM]     * max(1, n // 5)
        )
        types = (types * ((n // len(types)) + 1))[:n]
        resources = [Resource(idx, rt, self.cfg, rng) for idx, rt in enumerate(types)]
        LOG.info("Initialised %d resources: %s", n,
                 {_RES_NAMES[rt]: types.count(rt) for rt in set(types)})
        return resources

    def _spawn_wave(self, forced_size: Optional[int] = None) -> None:
        self.wave += 1
        rng = self.rng.get("patients")
        n   = forced_size or int(rng.integers(2, min(6, self.cfg.max_patients // 3) + 1))
        current = len([p for p in self.patients if p.active])
        n = min(n, self.cfg.max_patients - current)
        if n <= 0:
            LOG.debug("[WAVE %02d] Hospital at capacity", self.wave)
            return

        self.forecaster.record_wave(n)
        pred_mu, pred_std = self.forecaster.predict_next()
        surge_flag = " ⚠ SURGE DETECTED" if self.forecaster.is_surge() else ""
        self.events.appendleft(
            f"[WAVE {self.wave:02d}]  {n} admissions  "
            f"│ Forecast {pred_mu:.1f}±{pred_std:.1f}{surge_flag}")

        cat_weights = [0.30, 0.15, 0.30, 0.15, 0.10]
        cats = rng.choice(len(cat_weights), size=n, p=cat_weights)
        for cat_int in cats:
            cat = PatientCategory(cat_int)
            pat = Patient(self.cfg, cat, self.wave, rng)
            self.patients.append(pat)
            self.filters[pat.uid] = AcuityFilter(pat.acuity)

        LOG.info("[WAVE %02d] admitted %d patients  (forecast next: %.1f±%.1f%s)",
                 self.wave, n, pred_mu, pred_std, surge_flag)

    def _admit_readmission(self, cat_int: int, base_acuity: float) -> None:
        rng = self.rng.get("patients")
        cat = PatientCategory(cat_int)
        pat = Patient(self.cfg, cat, self.wave, rng,
                      initial_acuity=base_acuity, readmission=True)
        self.patients.append(pat)
        self.filters[pat.uid] = AcuityFilter(pat.acuity)
        self.readmit_count += 1
        self.events.appendleft(
            f"[READMIT T+{self.t:.0f}h] P{pat.uid:03d} {pat.name_str()} "
            f"acuity={base_acuity:.2f}")
        LOG.debug("[READMIT] %s  acuity=%.2f", pat, base_acuity)

    def _apply_night_shift(self) -> None:
        night = (self.step_n >= self.cfg.max_frames // 2)
        for res in self.resources:
            if res.rtype == ResourceType.STAFF_TEAM:
                if night:
                    res._night_capacity_override = max(1, int(res.capacity * 0.6))
                else:
                    res._night_capacity_override = None

    def tick(self) -> float:
        t0 = time.perf_counter()
        self.step_n += 1
        self.t      += self.cfg.dt
        rng = self.rng.get("step")

        if self.cfg.night_shift:
            self._apply_night_shift()

        for pat in self.patients:
            eff = pat.assigned_resource.effectiveness if pat.assigned_resource else 1.0
            pat.step(self.cfg.dt, rng, resource_effectiveness=eff)

        for pat in self.patients:
            if pat.active and pat.uid in self.filters:
                self.filters[pat.uid].predict(self.cfg.dt)
                self.filters[pat.uid].update(pat.acuity)

        for res in self.resources:
            res.step(self.cfg.dt, enable_fatigue=self.cfg.enable_fatigue)

        self.alloc.demand_pressure = self.forecaster.predict_pressure()
        self.alloc.sim_time        = self.t
        self.alloc.update_risk(self.patients, self.resources)

        if self.step_n % 3 == 0:
            self.alloc.assign(self.patients, self.resources, self.filters,
                              self.step_n, self.t)

        if self.step_n % 10 == 0:
            for pat in self.patients:
                if pat.active and not pat._sla_breached:
                    if self.sla.check(pat, self.t):
                        pat._sla_breached = True
                        for breach_msg in self.sla.latest_breaches(1):
                            self.events.appendleft(breach_msg)

        if self.readmissions is not None:
            for cat_int, acuity in self.readmissions.flush(self.step_n):
                current = len([p for p in self.patients if p.active])
                if current < self.cfg.max_patients:
                    self._admit_readmission(cat_int, acuity)

        still_active = []
        for pat in self.patients:
            if not pat.active:
                continue

            if pat.acuity >= MORTALITY_ACUITY:
                self.mortalities += 1
                res = pat.assigned_resource
                if res:
                    res.release(pat)
                pat.active = False
                self.filters.pop(pat.uid, None)
                if self.alloc._acpl_tick_state is not None:
                    self.acpl.feedback(
                        state=self.alloc._acpl_tick_state, action=0,
                        reward=-1.0, next_state=self.alloc._acpl_tick_state,
                        done=True, consequence=1.0)
                self.ward.accumulate_heat(*self._patient_xy(pat), value=2.0)
                evt = AllocationEvent(
                    tick=self.step_n, sim_time=self.t, outcome="mortality",
                    patient_uid=pat.uid, patient_category=pat.name_str(),
                    patient_wave=pat.wave,
                    resource_idx=res.idx if res else -1,
                    resource_type=res.name_str() if res else "none",
                    acuity_at_event=pat.acuity, wait_ticks=pat.ticks_waiting,
                    care_ticks=pat.ticks_in_care, occupancy_at_evt=self.alloc._occupancy,
                    acpl_penalty=self.acpl.last_mean_C, readmission=pat.readmission)
                self.metrics.push_event(evt)
                self.events.appendleft(
                    f"[MORTALITY T+{self.t:.0f}h] P{pat.uid:03d} {pat.name_str()}")
                continue

            if pat.is_dischargeable():
                self.discharges += 1
                res = pat.assigned_resource
                if res:
                    res.kills_total += 1
                    res.release(pat)
                pat.active = False
                self.filters.pop(pat.uid, None)
                if self.alloc._acpl_tick_state is not None:
                    self.acpl.feedback(
                        state=self.alloc._acpl_tick_state, action=1,
                        reward=+1.0, next_state=self.alloc._acpl_tick_state,
                        done=True, consequence=0.0)
                if self.readmissions is not None:
                    self.readmissions.schedule(pat, self.step_n)
                evt = AllocationEvent(
                    tick=self.step_n, sim_time=self.t, outcome="discharged",
                    patient_uid=pat.uid, patient_category=pat.name_str(),
                    patient_wave=pat.wave,
                    resource_idx=res.idx if res else -1,
                    resource_type=res.name_str() if res else "none",
                    acuity_at_event=pat.acuity, wait_ticks=pat.ticks_waiting,
                    care_ticks=pat.ticks_in_care, occupancy_at_evt=self.alloc._occupancy,
                    acpl_penalty=self.acpl.last_mean_C, readmission=pat.readmission)
                self.metrics.push_event(evt)
                continue

            still_active.append(pat)

        self.patients = still_active

        if self.step_n % self.cfg.admission_period == self.cfg.admission_period - 1:
            self._spawn_wave()

        active_n    = len([p for p in self.patients if p.active])
        total_slots = sum(r.capacity for r in self.resources)
        used_slots  = sum(r.used     for r in self.resources)
        occupancy   = used_slots / max(1, total_slots)
        self.occupancy_hist.append(occupancy * 100)
        self.patient_count_hist.append(active_n)
        total = self.discharges + self.mortalities
        self.discharge_rate_hist.append(self.discharges / max(1, total) * 100)
        self.mortality_hist.append(self.mortalities)
        avg_wait = float(np.mean([p.ticks_waiting for p in self.patients if p.active] or [0]))
        self.wait_time_hist.append(avg_wait)
        self.sla_violation_hist.append(self.sla.violation_rate() * 100)

        staff_res = [r for r in self.resources if r.rtype == ResourceType.STAFF_TEAM]
        avg_fat   = float(np.mean([r.fatigue for r in staff_res])) if staff_res else 0.0
        self.fatigue_hist.append(avg_fat * 100)

        if self.cfg.export_csv and self.step_n - self._last_csv_tick >= 300:
            tmp = self._csv_path + ".tmp"
            self.metrics.export_csv(tmp)
            try:
                os.replace(tmp, self._csv_path)
                self._last_csv_tick = self.step_n
            except OSError:
                pass

        if self.step_n % 100 == 0:
            dr = self.discharges / max(1, self.discharges + self.mortalities) * 100
            acpl_d = self.acpl.diagnostics()
            LOG.info(
                "[TICK %5d | T=%6.1fh | WAVE:%2d] "
                "Discharged=%-4d  Mortalities=%-3d  DR=%.1f%%  "
                "OccupancyPct=%.0f%%  ActivePts=%-3d | "
                "ACPL buf=%-4d λ=%.3f C=%.3f | "
                "SLA viol=%.1f%%  StaffFatigue=%.1f%%  Readmits=%d",
                self.step_n, self.t, self.wave,
                self.discharges, self.mortalities, dr,
                occupancy * 100, active_n,
                acpl_d["acpl_buf_size"], acpl_d["acpl_mean_lam"],
                acpl_d["acpl_mean_C"],
                self.sla.violation_rate() * 100, avg_fat * 100,
                self.readmit_count,
            )

        return (time.perf_counter() - t0) * 1000

    def _patient_xy(self, pat: Patient) -> Tuple[float, float]:
        idx = ([p.uid for p in self.patients].index(pat.uid)
               if pat in self.patients else 0)
        return self.ward.patient_display_pos(pat, idx % 25)


def run(cfg: ExperimentConfig):
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    sim = Simulation(cfg)
    G   = sim.ward.GRID

    plt.rcParams.update({
        "font.family":      "monospace",
        "axes.facecolor":   PAL["panel"],
        "figure.facecolor": PAL["bg"],
        "text.color":       PAL["white"],
        "axes.labelcolor":  PAL["dim"],
        "xtick.color":      PAL["dim"],
        "ytick.color":      PAL["dim"],
        "axes.edgecolor":   PAL["border"],
        "grid.color":       PAL["grid"],
        "grid.linewidth":   0.4,
        "axes.titlecolor":  PAL["green"],
        "toolbar":          "None",
        "figure.dpi":       80,
    })

    fig = plt.figure(figsize=(cfg.fig_width, cfg.fig_height))
    fig.patch.set_facecolor(PAL["bg"])
    fig.suptitle("INTELLIGENT HOSPITAL RESOURCE MANAGEMENT SYSTEM  ·  ACPL v2",
                 color=PAL["green"], fontsize=13, fontweight="bold", y=0.97)

    outer  = gridspec.GridSpec(1, 3, figure=fig,
                left=0.02, right=0.988, top=0.920, bottom=0.040,
                wspace=0.30, width_ratios=[2.2, 1.05, 1.08])
    left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                hspace=0.38, height_ratios=[2.8, 1.0])
    mid_gs  = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.68)
    rgt_gs  = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[2], hspace=0.70,
                height_ratios=[0.82, 1.08, 0.88, 1.35, 1.20])

    ax_ward = fig.add_subplot(left_gs[0])
    ax_ward.set_title("HOSPITAL WARD MAP  ·  LIVE PATIENT STATUS", fontsize=9, pad=4)
    ax_ward.set_xlim(0, G); ax_ward.set_ylim(0, G)
    ax_ward.set_aspect("equal"); ax_ward.axis("off")
    ax_ward.imshow(sim.ward.image, origin="lower", extent=[0,G,0,G], aspect="auto", zorder=0)
    for label, (cx, cy) in [("GENERAL WARD",(32,16)),("ICU",(51,10)),
                              ("A&E",(14,48)),("SURGICAL",(49,48)),("MATERNITY",(31,48))]:
        ax_ward.text(cx, cy, label, ha="center", va="center",
                     color=PAL["dim"], fontsize=6, alpha=0.7)
    _pat_scatter  = ax_ward.scatter([], [], s=80, zorder=5, edgecolors="white", linewidths=0.5)
    _crit_scatter = ax_ward.scatter([], [], s=140, marker="*", zorder=6,
                                    c=PAL["red"], alpha=0.9)
    _heatmap_img  = ax_ward.imshow(np.zeros((G, G)), origin="lower",
                                   extent=[0,G,0,G], aspect="auto",
                                   cmap="Reds", alpha=0.35, vmin=0, vmax=10, zorder=1)

    ax_log = fig.add_subplot(left_gs[1])
    ax_log.set_title("EVENT LOG", fontsize=8, pad=2)
    ax_log.axis("off")
    _log_texts = [ax_log.text(0.01, 1.0 - i*0.115, "",
                               transform=ax_log.transAxes, fontsize=6.5,
                               va="top", color=PAL["dim"]) for i in range(9)]

    ax_occ = fig.add_subplot(mid_gs[0])
    ax_occ.set_title("BED OCCUPANCY %", fontsize=8)
    ax_occ.set_ylim(0, 105); ax_occ.set_xlim(0, 500)
    ax_occ.axhline(80, color=PAL["yellow"], lw=0.7, alpha=0.6, ls="--")
    ax_occ.axhline(95, color=PAL["red"],    lw=0.7, alpha=0.6, ls="--")
    _occ_line, = ax_occ.plot([], [], color=PAL["cyan"], lw=1.2)

    ax_dr = fig.add_subplot(mid_gs[1])
    ax_dr.set_title("DISCHARGE RATE %", fontsize=8)
    ax_dr.set_ylim(0, 105); ax_dr.set_xlim(0, 500)
    _dr_line, = ax_dr.plot([], [], color=PAL["green"], lw=1.2)

    ax_pt = fig.add_subplot(mid_gs[2])
    ax_pt.set_title("ACTIVE PATIENTS", fontsize=8)
    ax_pt.set_ylim(0, cfg.max_patients + 2); ax_pt.set_xlim(0, 500)
    _pt_line, = ax_pt.plot([], [], color=PAL["yellow"], lw=1.2)

    ax_kpi = fig.add_subplot(rgt_gs[0])
    ax_kpi.axis("off")
    kpi_defs = [
        ("DISCHARGES", "0",   PAL["green"]),
        ("MORTALITIES","0",   PAL["red"]),
        ("DISCHARGE %","—",   PAL["cyan"]),
        ("SLA BREACH%","—%",  PAL["orange"]),
        ("READMITS",   "0",   PAL["purple"]),
    ]
    _kpi_labels = {}
    for idx_k, (kname, kval, kcol) in enumerate(kpi_defs):
        ax_kpi.text(0.02, 1.0 - idx_k*0.20, kname, transform=ax_kpi.transAxes,
                    fontsize=5.5, color=PAL["dim"])
        _kpi_labels[kname] = ax_kpi.text(0.55, 1.0 - idx_k*0.20, kval,
                                          transform=ax_kpi.transAxes,
                                          fontsize=7.5, color=kcol, fontweight="bold")

    ax_util = fig.add_subplot(rgt_gs[1])
    ax_util.set_title("RESOURCE UTIL. + FATIGUE", fontsize=8)
    ax_util.set_xlim(0, 1); ax_util.set_ylim(-0.5, len(sim.resources)-0.5)
    ax_util.set_facecolor(PAL["panel"])
    _util_bars = ax_util.barh(range(len(sim.resources)), [0.0]*len(sim.resources),
                               color=PAL["cyan"], alpha=0.7, height=0.6)
    _fat_bars  = ax_util.barh(range(len(sim.resources)), [0.0]*len(sim.resources),
                               color=PAL["orange"], alpha=0.35, height=0.3, left=0)

    ax_sla = fig.add_subplot(rgt_gs[2])
    ax_sla.set_title("SLA BREACH RATE %", fontsize=8)
    ax_sla.set_ylim(0, 30); ax_sla.set_xlim(0, 500)
    ax_sla.axhline(5, color=PAL["yellow"], lw=0.7, alpha=0.6, ls="--")
    ax_sla.axhline(15, color=PAL["red"], lw=0.7, alpha=0.6, ls="--")
    _sla_line, = ax_sla.plot([], [], color=PAL["orange"], lw=1.2)
    ax_sla.set_facecolor(PAL["panel"])

    ax_acpl = fig.add_subplot(rgt_gs[3])
    ax_acpl.axis("off")
    ax_acpl.set_title("ACPL v2 DIAGNOSTICS", fontsize=8)
    _acpl_texts = [ax_acpl.text(0.02, 1.0 - i*0.15, "",
                                 transform=ax_acpl.transAxes,
                                 fontsize=6.0, color=PAL["purple"]) for i in range(6)]

    ax_acu = fig.add_subplot(rgt_gs[4])
    ax_acu.set_title("PATIENT ACUITY DIST.", fontsize=8)
    ax_acu.set_xlim(0, 1); ax_acu.set_ylim(0, 8)
    ax_acu.set_facecolor(PAL["panel"])
    _acu_bars = ax_acu.bar(np.linspace(0.05, 0.95, 10), [0]*10,
                            width=0.09, color=PAL["yellow"], alpha=0.7)

    _title_txt = fig.text(0.50, 0.958, "", ha="center", va="center",
                          fontsize=9.5, color=PAL["cyan"], fontweight="bold")

    def _update(frame):
        tick_ms = sim.tick()

        active_pats = [p for p in sim.patients if p.active]
        positions, colors, crit_pos = [], [], []
        for idx_p, pat in enumerate(active_pats):
            x, y = sim.ward.patient_display_pos(pat, idx_p)
            positions.append((x, y))
            colors.append(_CAT_COLORS[pat.category])
            if pat.acuity >= CRITICAL_ACUITY:
                crit_pos.append((x, y))
        if positions:
            xs, ys = zip(*positions)
            _pat_scatter.set_offsets(np.column_stack([xs, ys]))
            _pat_scatter.set_color(colors)
        else:
            _pat_scatter.set_offsets(np.empty((0, 2)))
        if crit_pos:
            cxs, cys = zip(*crit_pos)
            _crit_scatter.set_offsets(np.column_stack([cxs, cys]))
        else:
            _crit_scatter.set_offsets(np.empty((0, 2)))

        hm = sim.ward.heatmap.copy()
        if hm.max() > 0:
            hm = hm / max(hm.max(), 1.0) * 10
        _heatmap_img.set_data(hm)

        evts = list(sim.events)
        for i, txt_obj in enumerate(_log_texts):
            if i < len(evts):
                e = evts[i]
                col = (PAL["red"]    if "MORTAL" in e or "SLA BREACH" in e else
                       PAL["orange"] if "READMIT" in e or "SURGE" in e     else
                       PAL["green"]  if "WAVE" in e                         else PAL["dim"])
                txt_obj.set_text(e[:62]); txt_obj.set_color(col)
            else:
                txt_obj.set_text("")

        def _upd_line(line, data, ax):
            if data:
                line.set_data(range(len(data)), list(data))
                ax.set_xlim(0, max(500, len(data)))

        _upd_line(_occ_line, sim.occupancy_hist,      ax_occ)
        _upd_line(_dr_line,  sim.discharge_rate_hist, ax_dr)
        _upd_line(_pt_line,  sim.patient_count_hist,  ax_pt)

        sla_data = list(sim.sla_violation_hist)
        if sla_data:
            _sla_line.set_data(range(len(sla_data)), sla_data)
            ax_sla.set_xlim(0, max(500, len(sla_data)))
            ax_sla.set_ylim(0, max(5, max(sla_data) * 1.2))

        total   = sim.discharges + sim.mortalities
        dr_pct  = sim.discharges / total * 100 if total else 0.0
        occ_now = list(sim.occupancy_hist)[-1] if sim.occupancy_hist else 0.0
        sla_pct = list(sim.sla_violation_hist)[-1] if sim.sla_violation_hist else 0.0
        _kpi_labels["DISCHARGES"].set_text(str(sim.discharges))
        _kpi_labels["MORTALITIES"].set_text(str(sim.mortalities))
        _kpi_labels["DISCHARGE %"].set_text(f"{dr_pct:.1f}%")
        _kpi_labels["SLA BREACH%"].set_text(f"{sla_pct:.1f}%")
        _kpi_labels["SLA BREACH%"].set_color(
            PAL["red"] if sla_pct > 15 else PAL["yellow"] if sla_pct > 5 else PAL["green"])
        _kpi_labels["READMITS"].set_text(str(sim.readmit_count))

        for bar_u, bar_f, res in zip(_util_bars, _fat_bars, sim.resources):
            util = res.used / max(1, res.capacity)
            bar_u.set_width(util)
            bar_u.set_color(PAL["red"] if util > 0.9 else
                            PAL["yellow"] if util > 0.7 else PAL["cyan"])
            bar_f.set_width(res.fatigue)

        ad = sim.acpl.diagnostics()
        fat_data = list(sim.fatigue_hist)
        avg_fat  = fat_data[-1] if fat_data else 0.0
        acpl_lines = [
            f"Updates  : {ad['acpl_updates']}",
            f"Buf size : {ad['acpl_buf_size']}",
            f"P-loss   : {ad['acpl_p_loss']:.4f}",
            f"Mean λ   : {ad['acpl_mean_lam']:.3f}",
            f"Mean C   : {ad['acpl_mean_C']:.3f}",
            f"StfFatigue: {avg_fat:.1f}%",
        ]
        for txt_obj, line in zip(_acpl_texts, acpl_lines):
            txt_obj.set_text(line)

        acuities = [p.acuity for p in sim.patients if p.active]
        if acuities:
            counts, _ = np.histogram(acuities, bins=10, range=(0, 1))
            for bar_obj, cnt in zip(_acu_bars, counts):
                bar_obj.set_height(cnt)
            ax_acu.set_ylim(0, max(2, max(counts) + 1))

        forecast_str = sim.forecaster.forecast_str()
        _title_txt.set_text(
            f"TICK {sim.step_n:5d}  |  T={sim.t:.1f}h  |  WAVE {sim.wave:02d}  |  "
            f"DR={dr_pct:.1f}%  |  OCC={occ_now:.0f}%  |  {forecast_str}  |  "
            f"ACTIVE={len(active_pats)}")

        return (_pat_scatter, _crit_scatter, _heatmap_img,
                _occ_line, _dr_line, _pt_line, _sla_line)

    ani = animation.FuncAnimation(
        fig, _update, frames=cfg.max_frames,
        interval=cfg.anim_interval, blit=False, repeat=False)

    def on_close(event):
        sim.acpl.stop()
        if cfg.export_csv:
            sim.metrics.export_csv(sim._csv_path)
        try:
            cat_summary = sim.metrics.per_category_summary()
            sla_summary = sim.sla.per_category_rates()
            report = {
                "session":         _SESSION_ID,
                "experiment_id":   cfg.experiment_id,
                "algorithm":       "ACPL-v2",
                "discharges":      sim.discharges,
                "mortalities":     sim.mortalities,
                "discharge_pct":   round(sim.discharges / max(1, sim.discharges + sim.mortalities) * 100, 1),
                "total_waves":     sim.wave,
                "acpl_updates":    sim.acpl._update_count,
                "readmissions":    sim.readmit_count,
                "sla_violations":  sim.sla.total_violations(),
                "sla_breach_rate": round(sim.sla.violation_rate() * 100, 2),
                "per_category":    cat_summary,
                "sla_per_category":sla_summary,
            }
            rp = Path(cfg.output_dir) / f"session_report_{cfg.experiment_id}.json"
            with open(rp, "w") as f:
                json.dump(report, f, indent=2)
            LOG.info("Session report: %s", rp)
        except OSError as e:
            LOG.warning("Could not write session report: %s", e)

    fig.canvas.mpl_connect("close_event", on_close)
    return fig, ani, sim


def _run_batch(cfg: ExperimentConfig) -> None:
    rows = []
    for run_i in range(cfg.batch_runs):
        run_seed = cfg.seed + run_i
        bc = ExperimentConfig(
            seed=run_seed, max_frames=cfg.max_frames,
            export_csv=False, output_dir=cfg.output_dir,
            batch_runs=1, enable_fatigue=cfg.enable_fatigue,
            enable_readmission=cfg.enable_readmission,
        )
        bc_sim = Simulation(bc)
        for _ in range(cfg.max_frames):
            bc_sim.tick()
        total = bc_sim.discharges + bc_sim.mortalities
        dr    = bc_sim.discharges / total * 100 if total else 0.0
        cat   = bc_sim.metrics.per_category_summary()
        row = {
            "run":         run_i,
            "seed":        run_seed,
            "discharges":  bc_sim.discharges,
            "mortalities": bc_sim.mortalities,
            "dr_pct":      round(dr, 1),
            "waves":       bc_sim.wave,
            "readmissions":bc_sim.readmit_count,
            "sla_breaches":bc_sim.sla.total_violations(),
        }
        for cat_name, vals in cat.items():
            row[f"dc_{cat_name.lower()[:4]}"] = vals["discharged"]
            row[f"mt_{cat_name.lower()[:4]}"] = vals["mortalities"]
        rows.append(row)
        LOG.info("[BATCH %d/%d] seed=%d  DR=%.1f%%  d=%d  m=%d  sla=%d  ra=%d",
                 run_i + 1, cfg.batch_runs, run_seed, dr,
                 bc_sim.discharges, bc_sim.mortalities,
                 bc_sim.sla.total_violations(), bc_sim.readmit_count)
        bc_sim.acpl.stop()

    if not rows:
        return
    out = Path(cfg.output_dir) / f"batch_summary_{cfg.experiment_id}.csv"
    fields = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    LOG.info("Batch complete — summary: %s", out)

    drs = [r["dr_pct"] for r in rows]
    n   = len(drs)
    mean_dr = float(np.mean(drs))
    std_dr  = float(np.std(drs))
    se = std_dr / math.sqrt(max(1, n))
    ci_lo = mean_dr - 1.96 * se
    ci_hi = mean_dr + 1.96 * se
    LOG.info("DR: mean=%.1f%%  std=%.1f%%  95%%CI=[%.1f%%, %.1f%%]  min=%.1f%%  max=%.1f%%",
             mean_dr, std_dr, ci_lo, ci_hi, float(np.min(drs)), float(np.max(drs)))
    slas = [r["sla_breaches"] for r in rows]
    LOG.info("SLA breaches: mean=%.1f  std=%.1f", float(np.mean(slas)), float(np.std(slas)))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Intelligent Hospital Resource Management System — ACPL v2",
        epilog="""
Examples:
  python hospital_rms_v2.py                         # default run
  python hospital_rms_v2.py --scenario surge        # mass-casualty surge
  python hospital_rms_v2.py --scenario pandemic     # high load, low staff
  python hospital_rms_v2.py --scenario nightshift   # night-shift capacity
  python hospital_rms_v2.py --scenario critical     # ICU pressure
  python hospital_rms_v2.py --batch 5 --no-csv      # 5 headless seeds
  python hospital_rms_v2.py --no-fatigue --no-readmission
        """
    )
    parser.add_argument("--seed",              type=int,   default=2025)
    parser.add_argument("--frames",            type=int,   default=2000)
    parser.add_argument("--interval",          type=int,   default=45)
    parser.add_argument("--no-csv",            action="store_true")
    parser.add_argument("--exp-id",            type=str,   default="")
    parser.add_argument("--output-dir",        type=str,   default="hrms_runs")
    parser.add_argument("--batch",             type=int,   default=1)
    parser.add_argument("--num-resources",     type=int,   default=10)
    parser.add_argument("--max-patients",      type=int,   default=16)
    parser.add_argument("--adm-period",        type=int,   default=90)
    parser.add_argument("--theme",             type=str,   default="dark",
                        choices=["dark","light"])
    parser.add_argument("--width",             type=float, default=26.0)
    parser.add_argument("--height",            type=float, default=14.0)
    parser.add_argument("--scenario",          type=str,   default="",
                        choices=["","surge","pandemic","routine","critical","nightshift"])
    parser.add_argument("--no-fatigue",        action="store_true")
    parser.add_argument("--no-readmission",    action="store_true")
    args = parser.parse_args()

    num_res  = args.num_resources
    max_pts  = args.max_patients
    adm_per  = args.adm_period
    sc_label = ""
    night    = False

    if args.scenario == "surge":
        num_res = 8;  max_pts = 24; adm_per = 50;  sc_label = "MASS-CASUALTY SURGE"
    elif args.scenario == "pandemic":
        num_res = 6;  max_pts = 30; adm_per = 40;  sc_label = "PANDEMIC OVERLOAD"
    elif args.scenario == "routine":
        num_res = 14; max_pts = 10; adm_per = 130; sc_label = "ROUTINE OPERATIONS"
    elif args.scenario == "critical":
        num_res = 10; max_pts = 20; adm_per = 60;  sc_label = "CRITICAL CARE PRESSURE"
    elif args.scenario == "nightshift":
        num_res = 9;  max_pts = 18; adm_per = 70
        sc_label = "NIGHT SHIFT OPERATIONS"; night = True

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _reinit_file_logger(args.output_dir)

    exp_id = args.exp_id if args.exp_id else str(uuid.uuid4())[:8]

    cfg = ExperimentConfig(
        seed             = args.seed,
        num_resources    = num_res,
        max_patients     = max_pts,
        admission_period = adm_per,
        max_frames       = args.frames,
        anim_interval    = args.interval,
        export_csv       = not args.no_csv,
        experiment_id    = exp_id,
        output_dir       = args.output_dir,
        fig_width        = args.width,
        fig_height       = args.height,
        theme            = args.theme,
        batch_runs       = args.batch,
        enable_fatigue   = not args.no_fatigue,
        enable_readmission = not args.no_readmission,
        night_shift      = night,
        scenario_label   = sc_label,
    )

    LOG.info("╔══════════════════════════════════════════════════════════════════╗")
    LOG.info("║     INTELLIGENT HOSPITAL RESOURCE MANAGEMENT SYSTEM (HRMS)      ║")
    LOG.info("║                 Algorithm: ACPL v2  by Shashank Dev             ║")
    LOG.info("╠══════════════════════════════════════════════════════════════════╣")
    LOG.info("║  EXP: %-8s  SEED:%-6d  FRAMES:%-5d                     ║",
             cfg.experiment_id, cfg.seed, cfg.max_frames)
    LOG.info("║  RESOURCES:%-3d  MAX_PATIENTS:%-3d  ADM_PERIOD:%-4d            ║",
             cfg.num_resources, cfg.max_patients, cfg.admission_period)
    LOG.info("║  STATE_DIM:14  HEAD:Dueling  N-STEP:3                          ║")
    LOG.info("║  FATIGUE:%-5s  READMISSION:%-5s  NIGHT_SHIFT:%-5s           ║",
             str(cfg.enable_fatigue), str(cfg.enable_readmission), str(cfg.night_shift))
    if sc_label:
        LOG.info("║  SCENARIO: %-52s  ║", sc_label)
    LOG.info("╚══════════════════════════════════════════════════════════════════╝")

    if cfg.batch_runs > 1:
        _run_batch(cfg)
        return

    fig, ani, sim = run(cfg)

    try:
        plt.show()
    except KeyboardInterrupt:
        LOG.info("Simulation terminated by user.")
        sim.acpl.stop()
        if cfg.export_csv:
            sim.metrics.export_csv(sim._csv_path)
        sys.exit(0)


if __name__ == "__main__":
    main()
