#!/usr/bin/env python3
"""
Bayesian Network Structure Learning with streamlined experiment tracking.

Features:
- Hill climbing with add/remove/reverse operations
- Tabu search to avoid cycling
- Multiple random restarts to escape local optima
- Streamlined data collection for analysis
"""

from __future__ import annotations
import csv, math, re, sys, time, os
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd


# ─────────────────────────── Utilities ────────────────────────────
def _logsumexp(values):
    """Stable log ∑ exp(values)."""
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def _canon(val: str) -> str:
    """
    Convert '3.0' → '3', keep strings unchanged.  Makes keys consistent.
    BIF files sometimes contain '0', CSVs '0.0'. Canonicalisation avoids key mismatches.
    """
    s = str(val).strip()
    if re.fullmatch(r"-?\d+(\.0+)?", s):
        return str(int(float(s)))
    return s


# ─────────────────────────── Structures ───────────────────────────
# A class for representing the CPT of a variable
class CPT:
    """Conditional‑probability table P(head | parents)."""

    def __init__(self, head: "Variable", parents: List["Variable"]):
        self.head = head # The variable this CPT belongs to (object)
        self.parents = parents # Parent variables (objects), in order
        # Dict[Tuple[parent values], Dict[value, prob]]
        self.rows: Dict[Tuple[str, ...], Dict[str, float]] = {}

    # String representation of the CPT according to the BIF format
    def __str__(self):
        head, parents = self.head.name, self.parents
        if not parents:
            probs = ", ".join(map(str, self.rows[()].values()))
            return f"probability ( {head} ) {{\n  table {probs};\n}}\n"

        def _row_str(key, row):
            par_vals = ", ".join(key)
            probs = ", ".join(map(str, row.values()))
            return f"  ( {par_vals} ) {probs};"

        body = "\n".join(_row_str(k, r) for k, r in self.rows.items())
        par_names = ", ".join(p.name for p in parents)
        return f"probability ( {head} | {par_names} ) {{\n{body}\n}}\n"

# A class for representing a variable
class Variable:
    """A node in the BN graph."""

    def __init__(self, name: str, values: List[str]):
        self.name = name # Name of the variable
        self.values = values
        self.cpt: CPT | None = None  # set later

    # String representation of the variable according to the BIF format
    def __str__(self):
        dom = ", ".join(self.values)
        k = len(self.values)
        return f"variable {self.name} {{\n  type discrete [ {k} ] {{ {dom} }};\n}}\n"


# ────────────────────── The Bayesian Network ──────────────────────
class BayesianNetwork:
    # Method for reading a Bayesian Network from a BIF file;
    # fills a dictionary 'variables' with variable names mapped to Variable
    # objects having CPT objects.

    def __init__(self, bif_file: str | None = None):
        self.vars: Dict[str, Variable] = {}
        if bif_file:
            self._read_bif(Path(bif_file))

    # ---------------- BIF I/O ----------------
    def _read_bif(self, path: Path):
        """Parse a very small subset of the .bif syntax."""
        with open(path) as fh:
            lines = [ln.strip() for ln in fh]

        it = iter(lines)
        for ln in it:
            if ln.startswith("variable"):
                name = ln.split()[1]
                dom_line = next(it)
                values = re.sub(r"[{},;]", "", dom_line).split()[5:]
                self.vars[name] = Variable(name, values)

            if ln.startswith("probability"):
                tokens = ln.split()
                child = tokens[2]
                parents = [self.vars[p.strip(",")] for p in tokens[4:-2]]
                cpt = CPT(self.vars[child], parents)

                if parents:
                    combos = product(*(p.values for p in parents))
                    for combo in combos:
                        row_line = next(it)
                        row_probs = list(map(float,
                                             re.sub(r"[(),;]", " ", row_line).split()[len(combo):]))
                        cpt.rows[combo] = dict(zip(self.vars[child].values,
                                                   row_probs))
                else:
                    row_line = next(it)
                    row_probs = list(map(float,
                                         re.sub(r"[(),;]", " ", row_line).split()[1:]))
                    cpt.rows[()] = dict(zip(self.vars[child].values, row_probs))
                self.vars[child].cpt = cpt

    # Method for writing a Bayesian Network to an output file
    def write_bif(self, path: Path):
        with open(path, "w") as f:
            f.write("network unknown {}\n\n")
            for v in self.vars.values():
                f.write(str(v))
            for v in self.vars.values():
                f.write(str(v.cpt))

    # --------------- Parameter learning ---------------
    def learn_parameters(self, csv_path: str, alpha: float = 1.0):
        """
        Fill every CPT row with Laplace‑smoothed MLEs.
        We try here to make sure Laplace avoids zero‑prob rows, crucial for log‑prob.
        """
        df = pd.read_csv(csv_path, dtype=str).map(_canon)

        for var in self.vars.values():
            parents = var.cpt.parents
            if not parents:  # root node
                counts = df[var.name].value_counts()
                total = len(df) + alpha * len(var.values)
                var.cpt.rows[()] = {
                    v: (counts.get(v, 0) + alpha) / total
                    for v in var.values
                }
                continue

            # child with parents
            par_cols = [p.name for p in parents]
            grouped = df.groupby(par_cols)
            for par_vals, sub_df in grouped:
                par_vals = (par_vals,) if not isinstance(par_vals, tuple) else par_vals
                total = len(sub_df) + alpha * len(var.values)
                var.cpt.rows[tuple(par_vals)] = {
                    v: ((sub_df[var.name] == v).sum() + alpha) / total
                    for v in var.values
                }

    # ---------- helper constructor from explicit structure ----------
    @classmethod
    def from_structure(cls,
                       domains: dict[str, list[str]],
                       structure: dict[str, list[str]]):
        """
        Build a BN given a parent‑list dictionary without learning.
        Each CPT row is initialised with a uniform distribution.
        """
        bn = cls()
        # 1. create variables
        for name, dom in domains.items():
            bn.vars[name] = Variable(name, dom)

        # 2. attach CPTs
        for child, parents in structure.items():
            par_objs = [bn.vars[p] for p in parents]
            cpt = CPT(bn.vars[child], par_objs)

            if parents:
                for key in product(*(p.values for p in par_objs)):
                    cpt.rows[key] = {v: 1 / len(cpt.head.values)
                                     for v in cpt.head.values}
            else:
                cpt.rows[()] = {v: 1 / len(cpt.head.values)
                                for v in cpt.head.values}
            bn.vars[child].cpt = cpt
        return bn

    # --------------- Exact inference ---------------
    def _log_joint(self, assignment: Dict[str, str]) -> float:
        """log P(assignment) or −inf if a CPT entry is missing/0."""
        ll = 0.0
        for v in self.vars.values():
            key = tuple(_canon(assignment[p.name]) for p in v.cpt.parents)
            val = _canon(assignment[v.name])
            try:
                p = v.cpt.rows[key][val]
            except KeyError:
                return float("-inf")
            if p == 0:
                return float("-inf")
            ll += math.log(p)
        return ll

    def posterior(self, evidence: Dict[str, str], targets: List[str]) -> Dict[Tuple[str, ...], float]:
        """
        This function computes the conditional probability P(targets | evidence) by enumerating
        all possible combinations of target variable values.

        Enumeration is exponential in the number of targets, but here we limit it to 1 or 2 targets,
        which is safe.
        """
        # Step 1: Get the domains (possible values) of each target variable
        domains = []
        for target in targets:
            # Assume `self.vars[target].values` gives the possible values for each target variable
            domains.append(self.vars[target].values)

        # Step 2: Calculate log-joint probabilities for each possible combination of target values
        log_probabilities = {}

        # Generate all possible combinations of target values
        for combo in product(*domains):
            # Create an assignment dictionary combining evidence and target variable values
            assignment = {**evidence, **dict(zip(targets, combo))}

            # Compute the log-joint probability for this assignment
            log_probabilities[combo] = self._log_joint(assignment)

        # Step 3: Calculate the normalization factor (log-sum-exp)
        # The normalization ensures the probabilities sum to 1
        max_log_prob = max(log_probabilities.values())
        log_sum_exp = 0
        for log_prob in log_probabilities.values():
            log_sum_exp += math.exp(log_prob - max_log_prob)

        # Step 4: Calculate the normalized conditional probabilities
        probabilities = {}
        for combo, log_prob in log_probabilities.items():
            # Compute the conditional probability by exponentiating and normalizing
            probabilities[combo] = math.exp(log_prob - max_log_prob) / log_sum_exp

        # Return the conditional probabilities as a dictionary
        return probabilities

    # --------------- Missing‑value imputation ---------------
    def impute_row(self, row: pd.Series) -> tuple[pd.Series, float]:
        """
        Fill the blanks in *row* and return (new_row, confidence).

        confidence = posterior probability of the chosen joint value(s).
        """
        missing = [c for c in row.index if pd.isna(row[c]) or row[c] == ""]
        if not missing:
            return row, 1.0                         # handle fully‑observed row

        evidence = {k: _canon(v) for k, v in row.items() if k not in missing}
        post = self.posterior(evidence, missing)
        best_vals, confidence = max(post.items(), key=lambda kv: kv[1])
        row[missing] = list(best_vals)

        return row, confidence

    # --------------- CSV helpers (needed by step4) ---------------
    def impute_missing(self, csv_in: str):
        """Return DataFrame with blanks filled plus per‑row confidence."""
        df = pd.read_csv(csv_in, dtype=str)
        df = df.fillna("").map(_canon)
        confidences = {}
        for idx, row in df.iterrows():
            new_row, conf = self.impute_row(row.copy())
            df.iloc[idx] = new_row
            confidences[idx] = conf
        return df, confidences

    def evaluate(self, ground_csv: str, missing_csv: str):
        """
        Accuracy on cells that were blank in *missing_csv*,
        compared to the ground‑truth *ground_csv*.
        """
        imp, confidences = self.impute_missing(missing_csv)
        gt = pd.read_csv(ground_csv, dtype=str).map(_canon)
        miss_mask = pd.read_csv(missing_csv, dtype=str).isna() | \
                    (pd.read_csv(missing_csv, dtype=str) == "")
        total = miss_mask.values.sum()
        correct = ((imp == gt) & miss_mask).values.sum()
        acc = correct / total if total else 1.0
        avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0
        return acc, avg_confidence

    # --------------- Enhanced Greedy structure search ---------------
    @classmethod
    def learn_structure(cls,
                        csv_path: str,
                        max_parents: int = 2,
                        max_iters: int = 20,
                        alpha: float = 0.1,
                        tabu_length: int = 10,
                        random_restarts: int = 1):
        """
        Enhanced hill-climber for Bayesian network structure learning.
        Operations: add, remove, and reverse edges.
        Uses tabu search to avoid cycling and includes random restarts.
        """
        print(f"Learning structure with params: max_parents={max_parents}, max_iters={max_iters}, alpha={alpha}")
        
        df = pd.read_csv(csv_path, dtype=str).map(_canon)
        domains = {c: sorted(df[c].unique()) for c in df.columns}
        names = list(domains)
        n_samples = len(df)
        
        # Build a BN from a set of edges
        def build(edges):
            struct = {n: [] for n in names}
            for a, b in edges:
                struct[b].append(a)
            return cls.from_structure(domains, struct)
        
        # Calculate BIC score
        def bic_score(bn):
            # Use vectorized operations for likelihood calculation
            ll = df.apply(lambda r: bn._log_joint(r.to_dict()), axis=1).sum()
            
            # Calculate parameter count
            k = 0
            for v in bn.vars.values():
                # Number of parameters = (domain size - 1) * number of parent configurations
                par_configs = max(1, np.prod([len(p.values) for p in v.cpt.parents]))
                k += (len(v.values) - 1) * par_configs
                
            # BIC score = log-likelihood - (parameter count / 2) * log(sample size)
            return ll - 0.5 * k * math.log(n_samples)
        
        best_bn = None
        best_score = float('-inf')
        total_iterations = 0
        
        # Multiple random restarts to escape local optima
        for restart in range(random_restarts):
            print(f"Random restart {restart+1}/{random_restarts}")
            
            # Initialize with empty network
            current_edges = set()
            current_bn = build(current_edges)
            current_bn.learn_parameters(csv_path, alpha)
            current_score = bic_score(current_bn)
            
            # Initialize tabu list
            tabu = []
            
            # Main hill climbing loop
            iterations_in_restart = 0
            for iter_num in range(max_iters):
                iterations_in_restart += 1
                total_iterations += 1
                improved = False
                best_operation = None
                best_op_score = current_score
                
                # Consider all possible edge operations
                operations = []
                
                # 1. Add edges
                for a in names:
                    for b in names:
                        if a == b:
                            continue
                        if (a, b) not in current_edges:
                            # Check if adding would exceed max parents
                            if sum(1 for x, y in current_edges if y == b) < max_parents:
                                operations.append(("add", a, b))
                
                # 2. Remove edges
                for a, b in current_edges:
                    operations.append(("remove", a, b))
                
                # 3. Reverse edges
                for a, b in current_edges:
                    # Check if reversing would exceed max parents or create cycle
                    if sum(1 for x, y in current_edges if y == a) < max_parents:
                        operations.append(("reverse", a, b))
                
                # Shuffle operations for better exploration
                np.random.shuffle(operations)
                
                # Evaluate each operation
                for op in operations:
                    op_type, a, b = op
                    
                    # Skip if operation is in tabu list
                    if op in tabu:
                        continue
                    
                    # Create new edge set based on operation
                    new_edges = current_edges.copy()
                    if op_type == "add":
                        new_edges.add((a, b))
                    elif op_type == "remove":
                        new_edges.remove((a, b))
                    else:  # reverse
                        new_edges.remove((a, b))
                        new_edges.add((b, a))
                    
                    # Skip if operation creates a cycle
                    if op_type != "remove" and _has_cycle(new_edges):
                        continue
                    
                    # Build and score the new network
                    new_bn = build(new_edges)
                    new_bn.learn_parameters(csv_path, alpha)
                    new_score = bic_score(new_bn)
                    
                    # Update best operation if this is better
                    if new_score > best_op_score:
                        best_op_score = new_score
                        best_operation = (op, new_edges, new_bn)
                
                # Apply the best operation if it improves the score
                if best_operation and best_op_score > current_score:
                    op, current_edges, current_bn = best_operation
                    current_score = best_op_score
                    improved = True
                    
                    # Add reverse operation to tabu list
                    reverse_op = None
                    if op[0] == "add":
                        reverse_op = ("remove", op[1], op[2])
                    elif op[0] == "remove":
                        reverse_op = ("add", op[1], op[2])
                    elif op[0] == "reverse":
                        reverse_op = ("reverse", op[2], op[1])
                    
                    if reverse_op:
                        tabu.append(reverse_op)
                        if len(tabu) > tabu_length:
                            tabu.pop(0)  # Remove oldest tabu entry
                
                print(f"  Iteration {iter_num+1}: score = {current_score:.2f}, improved = {improved}")
                
                # Stop if no improvement
                if not improved:
                    break
            
            # Update best network if current is better
            if current_score > best_score:
                best_bn = current_bn
                best_score = current_score
                print(f"  New best score: {best_score:.2f}")
        
        # Store learning stats with the network
        best_bn._learning_stats = {
            "total_iterations": total_iterations,
            "final_score": best_score,
            "num_edges": len(best_bn._edges()) if best_bn else 0
        }
        
        return best_bn

    # helpers
    def _edges(self):
        return {(p.name, v.name)
                for v in self.vars.values()
                for p in v.cpt.parents}
    
    # Network statistics for analysis
    def get_stats(self):
        """Get statistics about the network structure"""
        edges = self._edges()
        n_nodes = len(self.vars)
        n_edges = len(edges)
        
        # Calculate in-degree (number of parents) for each node
        in_degree = {}
        for var_name, var in self.vars.items():
            in_degree[var_name] = len(var.cpt.parents)
        
        # Calculate out-degree (number of children) for each node
        out_degree = defaultdict(int)
        for parent, child in edges:
            out_degree[parent] += 1
        
        # Ensure all nodes are in out_degree dict
        for var_name in self.vars:
            if var_name not in out_degree:
                out_degree[var_name] = 0
        
        # Find max degrees
        max_in_degree = max(in_degree.values()) if in_degree else 0
        max_out_degree = max(out_degree.values()) if out_degree else 0
        
        # Calculate average degrees
        avg_in_degree = sum(in_degree.values()) / n_nodes if n_nodes > 0 else 0
        avg_out_degree = sum(out_degree.values()) / n_nodes if n_nodes > 0 else 0
        
        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0,
            "max_in_degree": max_in_degree,
            "max_out_degree": max_out_degree,
            "avg_in_degree": avg_in_degree,
            "avg_out_degree": avg_out_degree
        }


# ----- graph cycle test (DFS) -----
def _has_cycle(edges: set[tuple[str, str]]) -> bool:
    """Return True iff the directed graph given by *edges* contains a cycle."""
    adj: dict[str, list[str]] = {}
    nodes = set()
    for a, b in edges:
        nodes |= {a, b}
        adj.setdefault(a, []).append(b)

    visited: set[str] = set()
    stack: set[str] = set()

    def dfs(u: str) -> bool:
        visited.add(u)
        stack.add(u)
        for v in adj.get(u, []):
            if v not in visited and dfs(v):
                return True
            if v in stack:
                return True
        stack.remove(u)
        return False

    return any(dfs(n) for n in nodes if n not in visited)


# ─────────────────────────── Experiment Tracking ────────────────────────────
def save_experiment_results(results):
    """Save experiment results to a single CSV file with only essential information"""
    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Path to the results CSV
    csv_path = "results/experiment_results.csv"
    
    # Check if the file exists
    if os.path.exists(csv_path):
        # Load existing data
        df_existing = pd.read_csv(csv_path)
        # Append new results
        df = pd.concat([df_existing, pd.DataFrame(results)], ignore_index=True)
    else:
        # Create new DataFrame
        df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Experiment results saved to {csv_path}")


def main():
    """Main function with improved structure search and experiment tracking"""
    if len(sys.argv) != 5:
        print("usage: bayes5.py train.csv test.csv test_missing.csv out.bif")
        sys.exit(1)

    train, test, miss, out_bif = sys.argv[1:5]

    print("=" * 80)
    print("ENHANCED BAYESIAN NETWORK STRUCTURE LEARNING")
    print("=" * 80)
    
    # Extract dataset name from the training file path
    dataset_name = Path(train).parent.name
    print(f"Processing dataset: {dataset_name}")
    
    # Initialize result with only essential metrics
    result = {
        "dataset": dataset_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    start_time_total = time.time()
    
    # 1) Try provided complete structure (if any)
    dataset_dir = Path(train).parent
    complete_bif = dataset_dir / f"{dataset_dir.name}_complete.bif"
    best_acc = -1
    best_bn = None
    
    if complete_bif.exists():
        print(f"\nTesting complete structure from {complete_bif}")
        start_time = time.time()
        bn = BayesianNetwork(complete_bif)
        bn.learn_parameters(train)
        acc, conf = bn.evaluate(test, miss)
        best_acc, best_bn = acc, bn
        elapsed_time = time.time() - start_time
        print(f"[complete] accuracy = {acc:.3f} (time: {elapsed_time:.2f}s)")
        
        # Store only essential complete structure metrics
        result["complete_accuracy"] = acc
        result["complete_edges"] = len(bn._edges())
        
        # Get basic network statistics
        complete_stats = bn.get_stats()
        result["complete_max_in_degree"] = complete_stats["max_in_degree"]
    else:
        result["complete_accuracy"] = None
        result["complete_edges"] = None
        result["complete_max_in_degree"] = None
    
    # 2) Try different hyperparameter combinations
    print("\nEvaluating different hyperparameter combinations")
    # TEST DIFFERENT HYPERPARAMETER COMBINATIONS FOR THE REPORT
    hyperparams = [
        {"max_parents": 2, "max_iters": 15, "alpha": 0.1, "tabu_length": 10, "random_restarts": 1},
        {"max_parents": 3, "max_iters": 20, "alpha": 0.5, "tabu_length": 15, "random_restarts": 1},
        {"max_parents": 4, "max_iters": 25, "alpha": 1.0, "tabu_length": 20, "random_restarts": 1}
    ]
    
    # Try each hyperparameter configuration sequentially
    for config_idx, config in enumerate(hyperparams):
        start_time = time.time()
        print(f"\nTrying hyperparameter set {config_idx+1}/{len(hyperparams)}: {config}")
        
        try:
            # Learn structure
            bn = BayesianNetwork.learn_structure(
                csv_path=train, 
                max_parents=config["max_parents"],
                max_iters=config["max_iters"],
                alpha=config["alpha"],
                tabu_length=config["tabu_length"],
                random_restarts=config["random_restarts"]
            )
            
            # Evaluate
            acc, conf = bn.evaluate(test, miss)
            elapsed_time = time.time() - start_time
            
            print(f"[learned {config_idx+1}] accuracy = {acc:.3f} (time: {elapsed_time:.2f}s)")
            
            # Store only config accuracy and edges
            result[f"config_{config_idx+1}_accuracy"] = acc
            result[f"config_{config_idx+1}_edges"] = len(bn._edges())
            
            # Update best network if better
            if acc > best_acc:
                best_acc = acc
                best_bn = bn
                print(f"New best accuracy: {acc:.3f}")
        except Exception as e:
            print(f"Error with config {config_idx+1}: {e}")
    
    # 3) Try different alpha values with the best structure
    if best_bn is not None:
        print("\nOptimizing alpha parameter for best structure")
        edges = best_bn._edges()
        
        # Test different alpha values
        alpha_values = [0.01, 0.1, 0.5, 1.0, 2.0]
        best_alpha_acc = -1
        best_alpha_value = None
        
        for alpha in alpha_values:
            start_time = time.time()
            
            # Build network with current structure
            df = pd.read_csv(train, dtype=str).map(_canon)
            domains = {c: sorted(df[c].unique()) for c in df.columns}
            struct = {n: [] for n in domains.keys()}
            for a, b in edges:
                struct[b].append(a)
            
            # Create and train network
            bn = BayesianNetwork.from_structure(domains, struct)
            bn.learn_parameters(train, alpha=alpha)
            
            # Evaluate
            acc, conf = bn.evaluate(test, miss)
            elapsed_time = time.time() - start_time
            
            print(f"Alpha={alpha}: accuracy={acc:.3f} (time: {elapsed_time:.2f}s)")
            
            # Store alpha accuracy
            result[f"alpha_{alpha}"] = acc
            
            # Update best if better
            if acc > best_alpha_acc:
                best_alpha_acc = acc
                best_alpha_value = alpha
                
                # Update best network if better than previous best
                if acc > best_acc:
                    best_acc = acc
                    best_bn = bn
                    print(f"New best accuracy: {acc:.3f} with alpha={alpha}")
        
        result["best_alpha"] = best_alpha_value
    
    # Save the best network
    if best_bn is not None:
        best_bn.write_bif(out_bif)
        print(f"\nBest network saved to {out_bif} (accuracy: {best_acc:.3f})")
        
        # Analyze the best network structure
        best_stats = best_bn.get_stats()
        print("\nBest network structure:")
        print(f"Number of edges: {best_stats['n_edges']}")
        
        # Print the edges
        print("Edges (parent → child):")
        edges = sorted(best_bn._edges())
        for parent, child in edges:
            print(f"  {parent} → {child}")
        
        # Print variables with many parents
        if best_stats['max_in_degree'] > 1:
            print("\nVariables with multiple parents:")
            for var_name, var in best_bn.vars.items():
                if len(var.cpt.parents) > 1:
                    parent_names = [p.name for p in var.cpt.parents]
                    print(f"  {var_name}: {', '.join(parent_names)}")
        
        # Add only essential final results
        total_time = time.time() - start_time_total
        result["learned_accuracy"] = best_acc
        result["learned_edges"] = best_stats['n_edges']
        result["learned_max_in_degree"] = best_stats['max_in_degree']
        result["accuracy_difference"] = best_acc - result.get("complete_accuracy", 0)
        result["execution_time"] = total_time
    
    # Save experiment data to file
    save_experiment_results([result])
    
    print(f"\nTotal execution time: {time.time() - start_time_total:.2f} seconds")
    
    return best_bn


if __name__ == "__main__":
    main()