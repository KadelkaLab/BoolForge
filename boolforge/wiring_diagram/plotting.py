#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import warnings


class WiringDiagramPlottingMixin:
    def plot_modular_structure(
        self,
        ax=None,
        show=True,
        node_labels: bool = True,
        max_nodes: int = 50,
        curviness: float = 0.25,
    ):
        """
        Plot the wiring diagram as a directed acyclic graph of strongly connected components.
    
        The wiring diagram is first condensed into its strongly connected components (SCCs),
        yielding a directed acyclic graph (DAG). Each node in the plot represents one SCC.
    
        The layout is hierarchical (top-to-bottom) using topological generations, making
        feed-forward structure visually apparent, while condensing feedback loops.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, a new figure and axis are created.
        show : bool, default=True
            Whether to call ``plt.show()`` after plotting.
        node_labels : bool, default=True
            Whether to label SCC nodes by their size (only shown for SCCs of size > 1).
        max_nodes : int, default=50
            If the number of SCCs exceeds this value, edges are sparsified to reduce clutter.
        curviness : float, default=0.25
            Curvature of edges spanning multiple layers (0 = straight).
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis containing the plot.
        """
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse, FancyArrowPatch
        except:
            raise ImportError(
                "Plotting requires matplotlib. "
                "Install it with: pip install matplotlib"
            )
        
        def _ellipse_boundary_point(x0, y0, x1, y1, a, b):
            """
            Intersection of ray from (x0,y0) toward (x1,y1)
            with ellipse centered at (x0,y0) with semi-axes a,b.
            """
            dx = x1 - x0
            dy = y1 - y0
            if dx == 0 and dy == 0:
                return x0, y0
        
            t = 1.0 / np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
            return x0 + t * dx, y0 + t * dy
    
        N = self.N
        I = self.I
    
        # ------------------------------------------------------------------
        # Build directed graph of the wiring diagram
        # ------------------------------------------------------------------
        g_full = nx.DiGraph()
        g_full.add_nodes_from(range(N))
    
        outdegrees = np.zeros(N, dtype=int)
        for target, regulators in enumerate(I):
            for r in regulators:
                g_full.add_edge(r, target)
                outdegrees[r] += 1
    
        # ------------------------------------------------------------------
        # Compute SCCs
        # ------------------------------------------------------------------
        sccs = list(nx.strongly_connected_components(g_full))
        scc_sizes = np.array([len(scc) for scc in sccs])
    
        scc_index = {}
        for i, scc in enumerate(sccs):
            for node in scc:
                scc_index[node] = i
                
        if len(sccs)<2:
            warnings.warn('No plot created. The network consists of a single SCC', UserWarning)
            return None
    
        # ------------------------------------------------------------------
        # Build SCC DAG
        # ------------------------------------------------------------------
        dag_edges = set()
        for u, v in g_full.edges():
            su, sv = scc_index[u], scc_index[v]
            if su != sv:
                dag_edges.add((su, sv))
    
        dag = nx.DiGraph(dag_edges)
    
        if dag.number_of_nodes() > max_nodes:
            dag = nx.DiGraph(
                (u, v) for (u, v) in dag_edges
                if scc_sizes[u] > 1 or scc_sizes[v] > 1
            )
    
        # ------------------------------------------------------------------
        # Node types
        # ------------------------------------------------------------------
        types = np.zeros(len(sccs), dtype=int)
    
        for i, scc in enumerate(sccs):
            if scc_sizes[i] > 1:
                types[i] = 2
            else:
                node = next(iter(scc))
                if (g_full.in_degree(node) == 0 or
                    (g_full.in_degree(node) == 1 and
                     list(g_full.predecessors(node))[0] == node)
                ):
                    types[i] = -1
                elif outdegrees[node] == 0:
                    types[i] = 1
                else:
                    types[i] = 0
    
        # ------------------------------------------------------------------
        # Hierarchical layout: initial placement by generations
        # ------------------------------------------------------------------
        pos = {}
        layers = []
        
        generations = list(nx.topological_generations(dag))
        #max_n_per_generation = max([len(gen) for gen in generations])
        #n_generations = len(generations)
        for layer, generation in enumerate(generations):
            gen = list(generation)
            layers.append(gen)
    
            k = len(gen)
            if k == 1:
                pos[gen[0]] = (0.0, -layer)
            else:
                xs = np.linspace(-0.5, 0.5, k)
                for x, node in zip(xs, gen):
                    pos[node] = (x, -layer)
    
        # ------------------------------------------------------------------
        # NEW: barycentric horizontal refinement (FFL fix)
        # ------------------------------------------------------------------
        for gen in layers[1:]:  # skip first layer
            for v in gen:
                preds = list(dag.predecessors(v))
                if preds:
                    x_mean = np.mean([pos[p][0] for p in preds])
                    y = pos[v][1]
                    pos[v] = (x_mean, y)
    
        # tiny deterministic jitter to avoid exact overlaps
        eps = 1e-3
        for i, v in enumerate(dag.nodes()):
            x, y = pos[v]
            pos[v] = (x + eps * (i % 7), y)
            
        # ------------------------------------------------------------------
        # Post-process: spread nodes within each layer to use full width
        # ------------------------------------------------------------------
        max_width = 3
        for gen in layers:
            xs = np.array([pos[v][0] for v in gen])
            if len(xs) <= 1:
                continue
        
            # Sort nodes by x
            order = np.argsort(xs)
            gen_sorted = [gen[i] for i in order]
        
            # Reassign evenly spaced x positions
            width = max(1.0, min(len(gen_sorted) / 3, max_width))
            new_xs = np.linspace(-width / 2, width / 2, len(gen_sorted))
        
            for v, x in zip(gen_sorted, new_xs):
                pos[v] = (x, pos[v][1])
        
        # ------------------------------------------------------------------
        # Vertical micro-staggering within layers (reduce edge overlap)
        # ------------------------------------------------------------------
        epsilon = 0.25  # vertical spacing scale
        
        for gen in layers:
            if len(gen) <= 3:
                continue
        
            # sort nodes left-to-right
            gen_sorted = sorted(gen, key=lambda v: pos[v][0])
        
            for i, v in enumerate(gen_sorted):
                x, y = pos[v]
                # pattern: middle, down, up, middle, down, up, ...
                offset = (i % 3) * epsilon
                pos[v] = (x, y + offset)
    
        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------
        xs = np.array([x for x, y in pos.values()])
        ys = np.array([y for x, y in pos.values()])
        
        x_span = xs.max() - xs.min()
        y_span = ys.max() - ys.min()
        
        target_aspect = 1.5   # width / height
        current_aspect = x_span / max(y_span, 1e-6)

        if current_aspect < target_aspect:
            scale = target_aspect / current_aspect
            for v, (x, y) in pos.items():
                pos[v] = (x * scale, y)
        
        fig_width  = max(6, 0.6 * x_span * scale)
        fig_height = max(6, 0.6 * abs(y_span))
        
        
        if ax is None:
            _, ax = plt.subplots(figsize=(fig_width, fig_height))
            
        color_map = {
            -1: "#eeeeee",
             0: "#ffcccc",
             1: "#eeeeee",
             2: "#ff9999",
        }
    
        labels = None
        if node_labels:
            labels = {
                n: 'SCC of size '+str(scc_sizes[n]) if scc_sizes[n] > 1
                else self.variables[list(sccs[n])[0]]
                for n in dag.nodes()
            }
    
    
        xs = [x for x, y in pos.values()]
        ys = [y for x, y in pos.values()]
        
        pad_x = 1.0
        pad_y = 1.0
        
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    
        texts = {}
        for n, (x, y) in pos.items():
            label = labels[n] if labels else ""
            texts[n] = ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=8,
                zorder=3
            )
        
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


        # ------------------------------------------------------------------
        # Fix horizontal overlaps using measured text widths
        # ------------------------------------------------------------------
        half_width = {}
        half_height = {}
        
        for n, text in texts.items():
            bbox = text.get_window_extent(renderer=renderer)
            inv = ax.transData.inverted()
            (x0, y0), (x1, y1) = inv.transform(
                [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)]
            )
        
            half_width[n]  = 0.65 * (x1 - x0)
            half_height[n] = 0.90 * (y1 - y0)
        
        
        min_gap = 0.1  # extra spacing between nodes
        
        for gen in layers:
            if len(gen) <= 1:
                continue
        
            # sort left-to-right
            gen_sorted = sorted(gen, key=lambda v: pos[v][0])
        
            x_cursor = pos[gen_sorted[0]][0]
            new_pos = {gen_sorted[0]: x_cursor}
        
            for prev, curr in zip(gen_sorted[:-1], gen_sorted[1:]):
                required = (
                    half_width[prev] +
                    half_width[curr] +
                    min_gap
                )
                x_cursor = max(pos[curr][0], x_cursor + required)
                new_pos[curr] = x_cursor
        
            # re-center layer
            center = np.mean(list(new_pos.values()))
            for v in gen_sorted:
                pos[v] = (new_pos[v] - center, pos[v][1])
        
        # ------------------------------------------------------------------
        # FINAL barycentric refinement (FFL alignment fix)
        # ------------------------------------------------------------------

        for gen in layers[1:]:
            for v in gen:
                preds = list(dag.predecessors(v))
                if preds:
                    x_mean = np.mean([pos[p][0] for p in preds])
                    #pos[v] = (x_mean, pos[v][1])
                    alpha = 0.1   # 0 = no barycentric, 1 = full snap
                    x_new = alpha * x_mean + (1 - alpha) * pos[v][0]
                    pos[v] = (x_new, pos[v][1])
        
        node_layer = {}
        for i, gen in enumerate(layers):
            for v in gen:
                node_layer[v] = i

        for t in texts.values():
            t.remove()
        
        ellipse_axes = {}   # <-- ADD THIS
        
        for n, (x, y) in pos.items():
            a = half_width[n]
            b = half_height[n]#max(half_height[n], 0.3 * (2 * a))  # since height = max(2b, 0.6w)
        
            ellipse_axes[n] = (a, b)
        
        for n, (x, y) in pos.items():
            a, b = ellipse_axes[n]
        
            ellipse = Ellipse(
                (x, y),
                width=2*a,
                height=2*b,
                facecolor=color_map[types[n]],
                edgecolor="black",
                zorder=2,
            )
            ax.add_patch(ellipse)


        for u, v in dag.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
        
            a0, b0 = ellipse_axes[u]
            a1, b1 = ellipse_axes[v]
        
            sx, sy = _ellipse_boundary_point(x0, y0, x1, y1, a0, b0)
            tx, ty = _ellipse_boundary_point(x1, y1, x0, y0, a1, b1)
        
            if node_layer[v] - node_layer[u] > 1:
                if node_layer[v] - node_layer[u] > 1 and curviness != 0.0:
                    sign = -1 if pos[u][0] + pos[v][0] > 0 else 1
                    conn = f"arc3,rad={sign * curviness}"
                else:
                    conn = "arc3"
            else:
                conn = "arc3"
                
            arrow = FancyArrowPatch(
                (sx, sy),
                (tx, ty),
                arrowstyle='-|>',
                mutation_scale=10,
                linewidth=1.2,
                color='black',
                connectionstyle=conn,   # curved or straight
                zorder=1,
            )
            ax.add_patch(arrow)
            
        #Draw text again, now at the correct positions
        for n, (x, y) in pos.items():
            ax.text(
                x, y, labels[n],
                ha="center", va="center",
                fontsize=7,
                zorder=4,   # on top
            )
    
        ax.set_autoscale_on(False)

        # --------------------------------------------------
        # final hard limits: guarantee everything is visible
        # --------------------------------------------------
        xs = []
        ys = []
        
        for n, (x, y) in pos.items():
            a, b = ellipse_axes[n]
            xs.extend([x - a, x + a])
            ys.extend([y - b, y + b])
        
        PAD_X = 0.5
        PAD_Y = 0.5
        
        ax.set_xlim(min(xs) - PAD_X, max(xs) + PAD_X)
        ax.set_ylim(min(ys) - PAD_Y, max(ys) + PAD_Y)
        
        ax.set_autoscale_on(False)

        ax.set_axis_off()
    
        if show:
            plt.show()
    
        return ax
    
    
    def plot(
        self,
        max_expanded_sccs: int = 4,
        min_scc_size: int = 2,
        show: bool = True,
        curviness: float = 0.25,
    ):
        """
        Plot an integrated overview of the wiring diagram.
    
        The plot consists of:
          1) A top panel showing the modular structure of the network as a DAG of
             strongly connected components (SCCs).
          2) Bottom panels showing the internal wiring of selected SCCs using a
             circular layout.
    
        By default, the largest SCCs of size >= ``min_scc_size`` are expanded,
        up to ``max_expanded_sccs``.
    
        Parameters
        ----------
        max_expanded_sccs : int, default=4
            Maximum number of SCCs to expand and show in detail.
        min_scc_size : int, default=2
            Minimum SCC size to be eligible for expansion.
        show : bool, default=True
            Whether to call ``plt.show()`` at the end.
        curviness : float, default=0.25
            Curvature of edges spanning multiple layers in the modular graph (0 = straight).
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        """
        
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError(
                "Plotting requires matplotlib. "
                "Install it with: pip install matplotlib"
            )
    
        N = self.N
        I = self.I
    
        # ------------------------------------------------------------
        # Build full directed graph
        # ------------------------------------------------------------
        g = nx.DiGraph()
        g.add_nodes_from(range(N))
        for target, regulators in enumerate(I):
            for r in regulators:
                g.add_edge(r, target)
    
        # ------------------------------------------------------------
        # Compute SCCs
        # ------------------------------------------------------------
        sccs = list(nx.strongly_connected_components(g))
        sccs = [sorted(scc) for scc in sccs]
    
        node_to_scc = {}
        for i, scc in enumerate(sccs):
            for v in scc:
                node_to_scc[v] = i
                
        G_scc = nx.DiGraph()
        G_scc.add_nodes_from(range(len(sccs)))
        
        for u, v in g.edges:
            su = node_to_scc[u]
            sv = node_to_scc[v]
            if su != sv:
                G_scc.add_edge(su, sv)
    
        # Select SCCs to expand
        expandable = [scc for scc in sccs if len(scc) >= min_scc_size]
        expandable.sort(key=len, reverse=True)
        expanded_sccs = expandable[:max_expanded_sccs]
    
        n_expanded = len(expanded_sccs)
    
        # ------------------------------------------------------------
        # Figure and GridSpec
        # ------------------------------------------------------------
        if n_expanded == 0: #if just showing the modular graph
            fig = plt.figure(figsize=(8, 4))
            gs = fig.add_gridspec(1, 1)
            ax_top = fig.add_subplot(gs[0, 0])
        elif len(sccs)==1: #if just showing the single SCC
            fig = plt.figure(figsize=(4 * n_expanded, 4))
            gs = fig.add_gridspec(1)
        else: #if showing both
            fig = plt.figure(figsize=(4 * n_expanded, 6))
            gs = fig.add_gridspec(
                2,
                n_expanded,
                height_ratios=[2.2, 1.5],
            )
            ax_top = fig.add_subplot(gs[0, :])
    
        # ------------------------------------------------------------
        # Top panel: modular structure
        # ------------------------------------------------------------
        if len(sccs)>1:
            self.plot_modular_structure(ax=ax_top, show=False,curviness=curviness)
            ax_top.set_title("Modular structure (DAG of SCCs)")
    
        # ------------------------------------------------------------
        # Bottom panels: internal SCC structure
        # ------------------------------------------------------------
        for j, scc in enumerate(expanded_sccs):
            if len(sccs)>1:
                ax = fig.add_subplot(gs[1, j])
            else:
                ax = fig.add_subplot(gs[j])
            
            C = set(scc)

            # direct external inputs only
            inputs = {
                u
                for v in C
                for u in g.predecessors(v)
                if u not in C
            }
            
            nodes_local = C | inputs
            subg = g.subgraph(nodes_local).copy()
    
            for u, v in list(subg.edges):
                if u not in nodes_local or v not in C:
                    subg.remove_edge(u, v)
    
            #subg = g.subgraph(scc).copy()
    
            pos = {}

            epsilon = 0.35  # vertical spacing scale
            
            # Inputs on top
            inputs = sorted(inputs)
            k_in = len(inputs)
            for j, v in enumerate(inputs):
                offset = 0 if k_in <= 3 else (j % 3 - 1) * epsilon
                pos[v] = ((j - (k_in - 1) / 2) /k_in*2, 2.0 + offset)
            
            # SCC nodes in a circle below
            pos_scc = nx.circular_layout(C, scale=1.0, center=(0.0, 0.0))
            pos.update(pos_scc)   
            
            # Color nodes: all feedback (same SCC)
            node_colors = [
                "#ff9999" if v in C else "#eeeeee"
                for v in subg.nodes
            ]
            
            nx.draw_networkx(
                subg,
                pos=pos,
                ax=ax,
                node_color=node_colors,
                node_size=200,
                labels={v: self.variables[v] for v in subg.nodes()},
                font_size=9,
                with_labels=True,
            )
    
            ax.set_title(f"SCC of size {len(scc)}")
            ax.set_axis_off()
    
        fig.tight_layout()
    
        if show:
            plt.show()
    
        return fig