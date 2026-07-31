#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

def _compress_with_known_cycle(traj, cycle_len):
    len_traj = len(traj)
    best_trajectory = []
    best_cycle_len = -1
    best_length = math.inf
    for s in range(len_traj):
        for p in range(1, min(cycle_len, len_traj - s) + 1):
            proposed_period = traj[s : s + p]
            good_proposal = True
            for i in range(s, len_traj):
                if traj[i] != proposed_period[(i - s) % p]:
                    good_proposal = False
                    break
            if not good_proposal:
                continue
            
            len_proposal = s + p
            if len_proposal < best_length:
                best_length = len_proposal
                best_trajectory = traj[:s] + proposed_period
                best_cycle_len = p
    return best_trajectory, best_cycle_len