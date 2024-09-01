# -*- coding: utf-8 -*-
from UCI.utils.accountant import compute_rdp_G, get_privacy_spent

def epsilon(N, batch_size, noise_multiplier, iterations, delta=1e-5):
    q = batch_size / N
    optimal_order = ternary_search(apply_dp_sgd_analysis, q, noise_multiplier, iterations, delta, 1, 512, 72)
    return apply_dp_sgd_analysis(q, noise_multiplier, iterations, [optimal_order], delta)

def apply_dp_sgd_analysis(q, sigma, iterations, orders, delta):
    rdp = compute_rdp_G(q, sigma, iterations, orders)
    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    return eps

def ternary_search(f, q, sigma, iterations, delta, left, right, iterations_ternary):
    for _ in range(iterations_ternary):
        left_third = left + (right - left) / 3
        right_third = right - (right - left) / 3
        if f(q, sigma, iterations, [left_third], delta) < f(q, sigma, iterations, [right_third], delta):
            right = right_third
        else:
            left = left_third
    return (left + right) / 2

# # 使用示例
# epsilon_value = epsilon(2300, 64, 0.0005, 700, 1e-5)
#
# print("Epsilon:", epsilon_value)
