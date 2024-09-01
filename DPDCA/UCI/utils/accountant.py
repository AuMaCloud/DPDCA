# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RDP analysis of the Sampled Gaussian Mechanism.

Functionality for computing Renyi differential privacy (RDP) of an additive
Sampled Gaussian Mechanism (SGM). Its public interface consists of two methods:
  compute_rdp(q, noise_multiplier, T, orders) computes RDP for SGM iterated
                                   T times.
  get_privacy_spent(orders, rdp, target_eps, target_delta) computes delta
                                   (or eps) given RDP at multiple orders and
                                   a target value for eps (or delta).

Example use:

Suppose that we have run an SGM applied to a function with l2-sensitivity 1.
Its parameters are given as a list of tuples (q1, sigma1, T1), ...,
(qk, sigma_k, Tk), and we wish to compute eps for a given delta.
The example code would be:

  max_order = 32
  orders = range(2, max_order + 1)
  rdp = np.zeros_like(orders, dtype=float)
  for q, sigma, T in parameters:
   rdp += rdp_accountant.compute_rdp(q, sigma, T, orders)
  eps, _, opt_order = rdp_accountant.get_privacy_spent(rdp, target_delta=delta)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import numpy as np


from scipy import special


########################
# LOG-SPACE ARITHMETIC #
########################

# 在对数空间中执行加法操作
def log_add(logx, logy):
    """Add two numbers in the log space."""
    if logx == -np.inf:
        return logy
    if logy == -np.inf:
        return logx

    max_log, min_log = max(logx, logy), min(logx, logy)
    return max_log + math.log1p(math.exp(min_log - max_log))

# 在对数空间中执行减法操作
def log_sub(logx, logy):
    """Subtract two numbers in the log space. Answer must be non-negative."""
    if logy == -np.inf:  # 减去0，结果是自身
        return logx
    if logx == logy:  # 减去相等的数，结果是负无穷
        return -np.inf
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")

    return math.log(math.expm1(logx - logy)) + logy  # 使用expm1来减少数值误差

# 将对数值格式化为字符串
def log_print(logx):
    """Pretty print."""
    if logx < math.log(sys.float_info.max):
        return str(math.exp(logx))
    else:
        return f"exp({logx})"

# 计算整数 alpha 对应的 log(alpha)
def compute_log_a_int(q, sigma, alpha):
    """Compute log(alpha) for integer alpha. 0 < q < 1."""
    assert isinstance(alpha, int)

    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
        log_a = log_add(log_a, math.log(special.binom(alpha, i)) + i * math.log(q) + (alpha - i) * math.log(1 - q) +
                    (i * i - i) / (2 * (sigma ** 2)))

    return float(log_a)

# 计算分数 alpha 对应的 log(alpha)
def compute_log_a_frac(q, sigma, alpha):
    log_a0, log_a1 = -np.inf, -np.inf
    i, z0 = 0, sigma ** 2 * math.log(1 / q - 1) + .5

    while True:
        coef = special.binom(alpha, i)
        log_coef, j = math.log(abs(coef)), alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(.5) + log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(.5) + log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = log_add(log_a0, log_s0)
            log_a1 = log_add(log_a1, log_s1)
        else:
            log_a0 = log_add(log_a0, log_s0)
            log_a1 = log_add(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return log_add(log_a0, log_a1)

# 计算任何正有限 alpha 对应的 log(A_alpha)
def compute_log_a(q, sigma, alpha):
    """Compute log(alpha) for any positive finite alpha."""
    if float(alpha).is_integer():
        return compute_log_a_int(q, sigma, int(alpha))
    else:
        return compute_log_a_frac(q, sigma, alpha)

# 计算 log(erfc(x)) 的对数空间函数
def log_erfc(x):
    try:
        return math.log(2) + special.log_ndtr(-x * 2**0.5)
    except NameError:
        r = special.erfc(x)
        if r == 0.0:
            # Approximation of log(erfc(x)) for large x
            return (-math.log(math.pi) / 2 - math.log(x) - x**2 - 0.5 * x**-2 +
                    0.625 * x**-4 - 37. / 24. * x**-6 + 353. / 64. * x**-8)
        else:
            return math.log(r)

# 根据 RDP 值计算 delta
def compute_delta(orders, rdp, eps):
    orders_vec, rdp_vec = np.atleast_1d(orders, rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")

    deltas = np.exp((rdp_vec - eps) * (orders_vec - 1))
    idx_opt = np.argmin(deltas)
    return min(deltas[idx_opt], 1.), orders_vec[idx_opt]

# 根据 RDP 值计算 epsilon
def compute_eps(orders, rdp, delta):
    orders_vec, rdp_vec = np.atleast_1d(orders, rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")

    eps = rdp_vec - math.log(delta) / (orders_vec - 1)

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    return eps[idx_opt], orders_vec[idx_opt]

# 计算给定采样率、噪声标准差和阶数的 RDP
def compute_rdp(q, sigma, alpha):
    """Compute RDP of the Sampled Gaussian mechanism at order alpha."""
    if q == 0:
        return 0
    if q == 1.:
        return alpha / (2 * sigma ** 2)
    if np.isinf(alpha):
        return np.inf
    return compute_log_a(q, sigma, alpha) / (alpha - 1)

# 计算 Sampled Gaussian 机制的 RDP
def compute_rdp_G(q, noise_multiplier, steps, orders):
    """Compute RDP of the Sampled Gaussian Mechanism."""
    if np.isscalar(orders):
        rdp = compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array([compute_rdp(q, noise_multiplier, order)
                        for order in orders])

    return rdp * steps

# 计算 delta 或 epsilon
def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
    """Compute delta (or eps) from RDP values."""
    if target_eps is None and target_delta is None:
        raise ValueError("Exactly one of eps and delta must be None. (Both are).")

    if target_eps is not None and target_delta is not None:
        raise ValueError("Exactly one of eps and delta must be None. (None is).")

    if target_eps is not None:
        delta, opt_order = compute_delta(orders, rdp, target_eps)
        return target_eps, delta, opt_order
    else:
        eps, opt_order = compute_eps(orders, rdp, target_delta)
        return eps, target_delta, opt_order

# 从隐私账本计算 Sampled Gaussian 机制的 RDP
def compute_rdp_from_ledger(ledger, orders):
    """Compute RDP of Sampled Gaussian Mechanism from ledger."""
    total_rdp = 0
    for sample in ledger:
        effective_z = sum([(q.noise_stddev / q.l2_norm_bound) ** -2 for q in sample.queries]) ** -0.5
        total_rdp += compute_rdp(sample.selection_probability, effective_z, 1, orders)
    return total_rdp
