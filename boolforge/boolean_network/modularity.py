#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:52:32 2026

@author: ckadelka
"""


import math

from collections.abc import Sequence
from itertools import product
import numpy as np
import networkx as nx

from .. import utils
from ..modularity import compress_trajectories

  
class BooleanNetworkModularityMixin:
    def get_attractors_synchronous_exact_exploiting_modularity(self):
        sccs = self.get_strongly_connected_components()
        module_per_node = {}
        for i,scc in enumerate(sccs):
            for el in scc:
                module_per_node.update({el:i})
        n_modules = len(sccs)
        dag = self.get_modular_structure()
        G = nx.from_edgelist(dag,nx.DiGraph)
        modules_sorted = list(nx.topological_sort(G))
        nodes_in_each_module = [np.sort(list(sccs[module_id])) for module_id in range(n_modules)]
        pos_in_each_module = [dict(zip(nodes_in_each_module[module_id],range(len(nodes_in_each_module[module_id])))) for module_id in range(n_modules)]
        inputs_per_module = [[] for i in range(n_modules)]
        for (a,b) in dag:
            inputs_per_module[b].append(a)
        
        module_attractors_binary = [[] for i in range(n_modules)]
        for module_id in modules_sorted:
            
            if len(inputs_per_module[module_id])==0:
                F1 = [self.F[j] for j in list(sccs[module_id])]
                I1 = [self.I[j].copy() for j in list(sccs[module_id])] 
                #reindex I1 to keep track of which nodes are in the module
                nodes_in_module = nodes_in_each_module[module_id]
                pos_in_module = pos_in_each_module[module_id]
                for i in range(len(nodes_in_module)):
                    for j in range(len(I1[i])):
                        I1[i][j] = pos_in_module[I1[i][j]]
                bn_of_module = self.__class__.BooleanNetwork(F=F1, I=I1)
                module_attractors = bn_of_module.get_attractors_synchronous_exact()['Attractors']
                module_attractors_binary[module_id] = [
                    np.array(list(
                        map(lambda x: 
                            utils.dec2bin(x,bn_of_module.N),
                            module_attractors[i]
                        )
                    ))
                    for i in range(len(module_attractors))
                ]
            else:  
                nodes_in_module = nodes_in_each_module[module_id]
                pos_in_module = pos_in_each_module[module_id]
                n_nodes_in_module = len(pos_in_module)
               
                #Getting all the nodes for every input module
                input_nodes = {}
                count = 0
                for node in nodes_in_module:
                    for regulator in self.I[node]:
                        try:
                            pos_in_module[regulator]
                        except KeyError:
                            try:
                                input_nodes[regulator]
                            except KeyError:
                                input_nodes.update({regulator:n_nodes_in_module+count})
                                count+=1
                
                pos_in_module.update(input_nodes)
                
                
                pos_in_module_per_input_nodes = {}
                for input_node in input_nodes.keys():
                    pos_in_module_per_input_nodes.update({input_node: pos_in_each_module[module_per_node[input_node]][input_node]})
                
                nodes_per_upstream_module_we_care_about = {}
                for input_node in input_nodes.keys():
                    upstream_module_id = module_per_node[input_node]
                    try:
                        nodes_per_upstream_module_we_care_about[upstream_module_id].add(input_node)
                    except KeyError:
                        nodes_per_upstream_module_we_care_about.update({upstream_module_id:set([input_node])})
                        
                
                
                #Getting the Upstream Attractors
                upstream_attractors = []
                for upstream_module in inputs_per_module[module_id]:
                    upstream_attractors.append(module_attractors_binary[upstream_module])
                    
                all_input_patterns = []
                for attractor_of_each_upstream_module in product(*upstream_attractors):
                    all_input_patterns.append([])
                    for attractor,upstream_module_id in zip(attractor_of_each_upstream_module,inputs_per_module[module_id]):
                        all_input_patterns[-1].append(utils.flatten(attractor[:,np.array(list(map(lambda x: pos_in_module_per_input_nodes[x],nodes_per_upstream_module_we_care_about[upstream_module_id])))]))
    
                F2 = [self.F[j] for j in nodes_in_module]
                I2 = [list(map(lambda x: pos_in_module[x],self.I[j])) for j in nodes_in_module]                
                bn_of_module = self.__class__.BooleanNetwork(F=F2, I=I2)
                    
                module_attractors = bn_of_module.get_attractors_synchronous_exact_with_external_inputs(all_input_patterns[0])['Attractors']
                
                #next line needs to be checked
                module_attractors_binary[module_id] = [np.array(list(
                    map(lambda x: utils.dec2bin(x,len(nodes_in_module)),
                        module_attractors[i])
                    ))
                    for i in range(len(module_attractors))
                ]
        return module_attractors_binary


    def _compute_post_transient_states(
        self,
        transient_input_sequence
    ):
        """
        Apply the transient input sequence once to every state
        and return the resulting unique states.
        """
        N_identity_nodes = len(self.get_identity_nodes(False))
        N_regulated_nodes = self.N - N_identity_nodes
    
        if not transient_input_sequence:
            return list(range(2**N_regulated_nodes))
    
        max_len = max(len(seq) for seq in transient_input_sequence)
        num_inputs = len(transient_input_sequence)
    
        fixed_network_cache = {}
        resulting_states = set()
    
        for state in range(2**N_regulated_nodes):
    
            vec = utils.dec2bin(state, N_regulated_nodes)
    
            for t in range(max_len):
                values = [
                    transient_input_sequence[i][t]
                    for i in range(num_inputs)
                ]
    
                values_dec = utils.bin2dec(values)
    
                if values_dec not in fixed_network_cache:
                    fixed_network_cache[values_dec] = self.get_network_with_fixed_identity_nodes(values)
    
                vec = fixed_network_cache[values_dec].update_network_synchronously(vec)
    
            resulting_states.add(utils.bin2dec(vec))
    
        return list(resulting_states)


    def _format_attractors_as_binary(
        self,
        attractors
    ):
        """
        Convert attractors from (external_decimal, state_decimal)
        to concatenated binary vectors.
        """
    
        N_identity_nodes = len(self.get_identity_nodes(False))
        N_regulated_nodes = self.N - N_identity_nodes
    
        formatted = []
    
        for attr in attractors:
            formatted_attr = []
            for external_dec, state_dec in attr:
                if N_identity_nodes > 0:
                    ext_bits = utils.dec2bin(external_dec, N_identity_nodes)
                else:
                    ext_bits = []
                state_bits = utils.dec2bin(state_dec, N_regulated_nodes)
                formatted_attr.append(ext_bits + state_bits)
            formatted.append(formatted_attr)
        return formatted


    def get_attractors_synchronous_exact_non_autonomous(self,
        transient_input_sequence : Sequence[Sequence[int]],
        periodic_input_sequence : Sequence[Sequence[int]]) -> dict:
        """
        Compute all attractors and basin sizes under synchronous updating
        for a Boolean network driven by an external (non-autonomous) input
        sequence.
        
        The external input sequence is split into two parts:
        
        1. A transient input sequence, applied once.
        2. A periodic input sequence, repeated indefinitely.
        
        First, the transient input sequence is applied to every network state
        to determine the set of states that serve as initial conditions for
        the periodic regime. Then, attractors and basin sizes are computed
        exactly under the periodic input sequence.
        
        Parameters
        ----------
        transient_input_sequence : sequence of sequence of int
            External input values applied during the transient phase.
            Each inner sequence corresponds to one external (identity)
            node and contains binary values (0 or 1) indexed by time.
            All sequences must have the same length.
        
        periodic_input_sequence : sequence of sequence of int
            External input values defining the periodic regime.
            Each inner sequence corresponds to one external (identity)
            node and contains binary values (0 or 1) forming a repeating
            pattern. Each sequence may have different length; the
            periodic regime is determined by their least common multiple.
        
        Returns
        -------
        result : dict
            Dictionary with the following keys:
        
            - Attractors : list
                List of attractors. Each attractor is represented as a list
                of pairs (external_input_decimal, state_decimal) describing
                one full cycle in the augmented (input, state) space.
        
            - NumberOfAttractors : int
                Total number of unique attractors.
        
            - BasinSizes : list of int
                Number of initial states (after the transient phase) that
                converge to each attractor.
        
            - AttractorDict : dict
                Mapping from (external_input_decimal, state_decimal) to
                attractor index.
        
            - STG : dict
                State transition graph of the periodic regime, mapping
                (external_input_decimal, state_decimal) to its successor.
        
            - InitialStatesPeriodic : list of int
                Set of network states (decimal representation) obtained
                after applying the transient input sequence.
        
            - FormattedAttractors : list
                Attractors represented as binary vectors obtained by
                concatenating the external input bits and the network
                state bits at each point in the cycle.
        """
    
        #Apply transient block
        initial_states = self._compute_post_transient_states(
            transient_input_sequence
        )
    
        #Compute periodic attractors
        result = self.get_attractors_synchronous_exact_with_external_inputs(
            periodic_input_sequence,
            initial_states
        )
    
        #Format attractors
        result["InitialStatesPeriodic"] = initial_states
        result["FormattedAttractors"] = self._format_attractors_as_binary(
                                            result["Attractors"]
                                        )
    
        return result

    def _build_periodic_input_matrix(self, periodic_input_sequence):
        lengths = [len(p) for p in periodic_input_sequence]
        lcm = math.lcm(*lengths)
    
        periodic_inputs = np.zeros((lcm, len(periodic_input_sequence)), dtype=int)
    
        for i, pattern in enumerate(periodic_input_sequence):
            reps = lcm // len(pattern)
            periodic_inputs[:, i] = pattern * reps
    
        return periodic_inputs, lcm
    
    def _build_phase_transition_maps(self, periodic_inputs):
        N_regulated_nodes = self.N - len(self.get_identity_nodes(False))
    
        transition_maps = []
    
        for phase_values in periodic_inputs:
            fixed_net = self.get_network_with_fixed_identity_nodes(phase_values)
    
            phase_map = {}
            for state in range(2**N_regulated_nodes):
                next_state = utils.bin2dec(
                    fixed_net.update_network_synchronously(
                        utils.dec2bin(state, N_regulated_nodes)
                    )
                )
                phase_map[state] = next_state
    
            transition_maps.append(phase_map)
    
        return transition_maps
    
    def _explore_phase_state_space(self, transition_maps, periodic_inputs, starting_states_dec):
        lcm = len(periodic_inputs)
    
        attractors = []
        basin_sizes = []
        attractor_dict = {}
        stg = {}
        
        for phase in range(lcm):
            for state in starting_states_dec:
                path = []
                visited_local = {}
    
                while True:
    
                    key = (phase, state)
    
                    if key in attractor_dict:
                        idx = attractor_dict[key]
                        basin_sizes[idx] += 1
                        break
    
                    if key in visited_local:
                        cycle_start = visited_local[key]
                        cycle = path[cycle_start:]
                        idx = len(attractors)
    
                        attractors.append(cycle)
                        basin_sizes.append(1)
    
                        for k in cycle:
                            attractor_dict[k] = idx
                        break
    
                    visited_local[key] = len(path)
                    path.append(key)
    
                    next_state = transition_maps[phase][state]
                    next_phase = (phase + 1) % lcm
    
                    stg[(utils.bin2dec(periodic_inputs[phase]), state)] = (
                        utils.bin2dec(periodic_inputs[next_phase]),
                        next_state
                    )
    
                    phase = next_phase
                    state = next_state
    
        return attractors, basin_sizes, attractor_dict, stg    
    
    
    def get_attractors_synchronous_exact_with_external_inputs(
        self,
        periodic_input_sequence : Sequence[Sequence[int]],
        starting_states_dec : [Sequence[int], None] = None) -> dict:
        """
        Compute all attractors and basin sizes under synchronous updating
        for a Boolean network with periodic external inputs.
        
        The external inputs are treated as a periodic sequence. The state
        transition graph is constructed over the combined space of
        (network state, input phase), and attractors are detected exactly.
        
        Parameters
        ----------
        periodic_input_sequence : sequence of sequence of int
            External input values defining the periodic regime.
            Each inner sequence corresponds to one external (identity)
            node and contains binary values (0 or 1) forming a repeating
            pattern. Each sequence may have different length; the
            periodic regime is determined by their least common multiple.
        
        starting_states_dec : sequence of int, optional
            Optional list of initial network states in decimal representation.
            If None, all possible states are used.
        
        Returns
        -------
        result : dict
            Dictionary with the following keys:
        
            - Attractors : list
                List of attractors. Each attractor is a list of pairs
                (external_input_decimal, state_decimal) forming a cycle.
        
            - NumberOfAttractors : int
                Total number of unique attractors.
        
            - BasinSizes : list of int
                Number of initial states converging to each attractor.
        
            - AttractorDict : dict
                Mapping from (external_input_decimal, state_decimal)
                to attractor index.
        
            - STG : dict
                State transition graph mapping
                (external_input_decimal, state_decimal) to the next pair.
        """    
        N_regulated_nodes = self.N - len(self.get_identity_nodes(False))
    
        if starting_states_dec is None:
            starting_states_dec = list(range(2**N_regulated_nodes))
    
        periodic_inputs, lcm = self._build_periodic_input_matrix(periodic_input_sequence)
    
        transition_maps = self._build_phase_transition_maps(periodic_inputs)
    
        attractors, basin_sizes, attractor_dict, stg = \
            self._explore_phase_state_space(
                transition_maps,
                periodic_inputs,
                starting_states_dec
            )
    
        formatted_attractors = []
        for attr in attractors:
            formatted_attractors.append([
                (utils.bin2dec(periodic_inputs[phase]), state)
                for phase, state in attr
            ])
    
        return {
            "Attractors": formatted_attractors,
            "NumberOfAttractors": len(formatted_attractors),
            "BasinSizes": basin_sizes,
            "AttractorDict": attractor_dict,
            "STG": stg
        }

    def _get_fnet_(self,values,fixed_network_cache):
        values_dec = utils.bin2dec(values)
        if values_dec in fixed_network_cache:
            fixed_network = fixed_network_cache[values_dec]
        else:
            fixed_network = self.get_network_with_fixed_identity_nodes(values)
            fixed_network_cache[values_dec] = fixed_network
        return fixed_network

    def _calculate_trajectory(
            self,
            starting_state_dec,
            transient_input_sequence,
            periodic_input_sequence,
            N_regulated_nodes,
            fixed_network_cache
            ):
        trajectory = [starting_state_dec]
        latest_state = starting_state_dec
        
        # Compute the non-periodic component of the trajectory.
        len_np = len(transient_input_sequence)
        max_len_pattern = max(list(zip(map(len, transient_input_sequence))))[0]
        for idx in range(max_len_pattern):
            vals = [ transient_input_sequence[node][idx] for node in range(len_np) ]
            fixed_network = self._get_fnet_(vals,fixed_network_cache)
            latest_state = utils.bin2dec(
                fixed_network.update_network_synchronously(
                    utils.dec2bin(latest_state, N_regulated_nodes)
                )
            )
            trajectory.append(latest_state)
            
        # Compute the periodic component of the trajectory.
        len_p = len(periodic_input_sequence)
        lcm = math.lcm(*list(map(len, periodic_input_sequence)))
        idx_p = 0
        cycle_len = -1
        
        seen = {}  # (state, phase) -> index in traj_cyclic
        traj_cyclic = []
        idx_p = 0
        
        while True:
            phase = idx_p % lcm
            key = (latest_state, phase)
        
            if key in seen:
                # We found the cycle start
                cycle_start = seen[key]
                cycle_len = len(traj_cyclic) - cycle_start
                break
        
            seen[key] = len(traj_cyclic)
        
            vals = [
                periodic_input_sequence[node][phase]
                for node in range(len_p)
            ]
            fixed_network = self._get_fnet_(vals,fixed_network_cache)
        
            latest_state = utils.bin2dec(
                fixed_network.update_network_synchronously(
                    utils.dec2bin(latest_state, N_regulated_nodes)
                )
            )
        
            traj_cyclic.append(latest_state)
            idx_p += 1
        trajectory.extend(traj_cyclic[:cycle_start + cycle_len])
        #print(trajectory, traj_cyclic, traj_cyclic[:cycle_start + cycle_len])
        
        # Compress the trajectory's representation to be minimal.
        # That is, only the transient input sequence and a single
        # cycle of the periodic input sequence.
        len_traj = len(trajectory)
        best_trajectory = []
        best_cycle_len = -1
        best_length = math.inf
        for s in range(len_traj):
            for p in range(1, min(cycle_len, len_traj - s) + 1):
                proposed_period = trajectory[s : s + p]
                good_proposal = True
                for i in range(s, len_traj):
                    if trajectory[i] != proposed_period[(i - s) % p]:
                        good_proposal = False
                        break
                if not good_proposal:
                    continue
                
                len_proposal = s + p
                if len_proposal < best_length:
                    best_length = len_proposal
                    best_trajectory = trajectory[:s] + proposed_period
                    best_cycle_len = p
        #print(best_trajectory, best_cycle_len, "\n")
        
        # Return the compressed trajectory array and the length of the
        # periodic component.
        # Note that the periodic_input_sequence will ALWAYS be the last
        # cycle_len values in the array. The periodic_input_sequence
        # also correspond with the attractors of the network.
        return best_trajectory, best_cycle_len
    
    def get_trajectories(
        self,
        transient_input_sequence,
        periodic_input_sequence,
        merge_trajectories=True,
        starting_states_dec=None
    ):
        """
        Compute synchronous trajectories of the Boolean network given a 
        non-autonomous external input sequence.
        
        The external input is split into two phases:
        
        1. A transient input sequence, applied once.
        2. A periodic input sequence, repeated indefinitely.
        
        For each specified initial state, the transient input sequence is
        applied first. The resulting state then evolves under the periodic
        input sequence. Periodicity is detected in the augmented space
        (state, input phase), ensuring that cycles are identified correctly
        even when the same network state appears at different input phases.
        
        Each trajectory is returned in minimal form, consisting of:
        
        - A non-periodic prefix (possibly empty),
        - Followed by a single instance of the periodic cycle.
        
        Parameters
        ----------
        transient_input_sequence : sequence of sequence of int
            External input values applied during the transient phase.
            Each inner sequence corresponds to one external (identity)
            node and contains binary values (0 or 1) indexed by time.
            All sequences must have the same length.
        
        periodic_input_sequence : sequence of sequence of int
            External input values defining the periodic regime.
            Each inner sequence corresponds to one external (identity)
            node and contains binary values (0 or 1) forming a repeating
            pattern. The effective period is the least common multiple
            of the individual sequence lengths.
        
        merge_trajectories : bool, optional (default=True)
            If True, trajectories are merged into a directed graph
            representing the non-autonomous state space (with consistent
            merging across trajectories). If False, individual trajectories
            are returned.
        
        starting_states_dec : sequence of int or None, optional
            Decimal representations of initial network states.
            If None, all states in the full state space (size 2^N) are used.
        
        Returns
        -------
        result : networkx.DiGraph or list
            If merge_trajectories is True:
                A directed graph representing the merged trajectories,
                i.e., the state space of the non-autonomous system.
        
            If merge_trajectories is False:
                A list of tuples (trajectory, cycle_length), where
        
                - trajectory : list of int
                    Decimal representations of states, containing the
                    non-periodic prefix followed by exactly one full
                    cycle of the periodic component.
        
                - cycle_length : int
                    Length of the periodic component (the last
                    cycle_length entries of trajectory).
        """        
        N_identity_nodes = len(self.get_identity_nodes(False))
        N_regulated_nodes = self.N - N_identity_nodes
    
        # Validation
        assert len(transient_input_sequence) == len(periodic_input_sequence)
        assert len(transient_input_sequence) == N_identity_nodes
        assert all(len(seq) > 0 for seq in periodic_input_sequence)
    
        if starting_states_dec is None:
            starting_states_dec = list(range(2 ** N_regulated_nodes))
        else:
            starting_states_dec = list(set(starting_states_dec))
    
        fixed_network_cache = {}
        trajectories = []
    
        for state in starting_states_dec:
            trajectory, cycle_len = self._calculate_trajectory(
                state,
                transient_input_sequence,
                periodic_input_sequence,
                N_regulated_nodes,
                fixed_network_cache
            )        
            trajectories.append((trajectory, cycle_len))
    
        if merge_trajectories:
            return compress_trajectories(trajectories, N_regulated_nodes)
    
        return trajectories
    
    