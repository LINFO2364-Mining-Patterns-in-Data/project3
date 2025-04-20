import itertools
import re
import sys
from collections import defaultdict
import pandas as pd


# =============================== #
# CONDITIONAL PROBABILITY TABLES  #
# =============================== #
class CPT:
    def __init__(self, head, parents):
        self.head = head
        self.parents = parents
        self.entries = {}

    def __str__(self):
        comma = ", "
        if len(self.parents) == 0:
            return f"probability ( {self.head.name} ) {{" + "\n" \
                f"  table {comma.join ( map(str,self.entries[tuple()].values () ))};" + "\n" \
                f"}}" + "\n"
        else:
            return f"probability ( {self.head.name} | {comma.join ( [p.name for p in self.parents ] )} ) {{" + "\n" + \
                "\n".join ( [  \
                  f"  ({comma.join(names)}) {comma.join(map(str,values.values ()))};" \
                    for names,values in self.entries.items () \
                ] ) + "\n}\n" 

# =============================== #
#    VARIABLES REPRESENTATION     #
# =============================== #       
class Variable:
    def __init__(self, name, values):
        self.name = name
        self.values = values 
        self.cpt = None

    def __str__(self):
        comma = ", "
        return f"variable {self.name} {{" + "\n" \
             + f"  type discrete [ {len(self.values)} ] {{ {(comma.join(self.values))} }};" + "\n" \
             + f"}}" + "\n"
    
# =============================== #
#          NETWORK MODEL          #
# =============================== #
class BayesianNetwork:
    def __init__(self, input_file):
        with open(input_file) as f:
            lines = f.readlines()
        self.variables = {}
        for i in range(len(lines)):
            lines[i] = lines[i].lstrip().rstrip().replace('/', '-')
        i = 0
        while i < len(lines) and not lines[i].startswith("probability"):
            if lines[i].startswith("variable"):
                variable_name = lines[i].rstrip().split(' ')[1]
                i += 1
                variable_def = lines[i].rstrip().split(' ')
                assert(variable_def[1] == 'discrete')
                variable_values = [x for x in variable_def[6:-1]]
                for j in range(len(variable_values)):
                    variable_values[j] = re.sub(r'\(|\)|,', '', variable_values[j])
                variable = Variable(variable_name, variable_values)
                self.variables[variable_name] = variable
            i += 1
        while i < len(lines):
            if lines[i].startswith('probability'):
                split = lines[i].split(' ')
                target_variable_name = split[2]
                variable = self.variables[target_variable_name]
                parents = [self.variables[x.rstrip().lstrip().replace(',', '')] for x in split[4:-2]]
                assert(variable.name == split[2])
                cpt = CPT(variable, parents)
                i += 1
                if len(parents) > 0:
                    nb_lines = 1
                    for p in parents:
                        nb_lines *= len(p.values)
                    for _ in range(nb_lines):
                        cpt_line = [x for x in re.sub(r'\(|\)|,', '', lines[i][:-1]).split()]
                        parent_values = tuple([x for x in cpt_line[:len(parents)]])
                        probabilities = [float(p) for p in cpt_line[len(parents):]]
                        cpt.entries[parent_values] = { v:p for v,p in zip(variable.values,probabilities) }
                        i += 1
                else:
                    cpt_line = [x for x in re.sub(r'\(|\)|,', '', lines[i][:-1]).split()]
                    probabilities = [float(p) for p in cpt_line[1:]]
                    cpt.entries[tuple()] = { v:p for v,p in zip(variable.values,probabilities) }
                variable.cpt = cpt
            i += 1

    def write(self,filename):
        with open(filename,"w") as file:
            for var in self.variables.values ():
                file.write(str(var))
            for var in self.variables.values ():
                file.write(str(var.cpt))

    def P_Yisy_given_parents_x(self,Y,y,x=tuple()):
        return self.variables[Y].cpt.entries[x][y]

    def P_Yisy_given_parents(self,Y,y,pa={}):
        x = tuple([ pa[parent.name] for parent in self.variables[Y].cpt.parents ])
        return self.P_Yisy_given_parents_x(Y,y,x)
    
    def _get_children(self):
        """
        Builds a mapping from each variable to its children in the Bayesian Network.

        Returns:
            dict: A dictionary where each key is a variable name and the value is a list of its children.
        """
        children = defaultdict(list)
        for var in self.variables.values():
            for parent in var.cpt.parents:
                children[parent.name].append(var.name)
        return children

    def _normalize(self, dist):
        """
        Normalizes a probability distribution so that the values sum to 1.

        Args:
            dist (dict): A dictionary mapping values to unnormalized probabilities.

        Returns:
            dict: The normalized distribution, or an empty dict if the total is 0.
        """
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items() if total > 0}

    def _get_pi_contribution(self, pname, pval, pi_msgs, evidence):
        """
        Computes the π-contribution of a parent variable for belief propagation.

        Args:
            pname (str): The name of the parent variable.
            pval (str): The value of the parent variable being considered.
            pi_msgs (dict): Dictionary of π-messages already computed.
            evidence (dict): Observed values for some variables.

        Returns:
            float: The π-contribution for the given parent value.
        """
        if pname in pi_msgs:
            return pi_msgs[pname][pval]
        elif pname in evidence:
            return 1.0 if evidence[pname] == pval else 0.0
        else:
            return 1.0 / len(self.variables[pname].values)
        
    def _get_vals(self, pname, xi=None, focus_node=None, evidence=None):
        """
        Returns possible values for a parent variable during message passing.

        Args:
            pname (str): The name of the parent variable.
            xi (str, optional): The current value of the focus node (if applicable).
            focus_node (str, optional): The node currently receiving messages.
            evidence (dict): Observed evidence.

        Returns:
            list: List of possible values for pname, based on evidence and context.
        """
        if evidence is None:
            evidence = {}
        if pname == focus_node:
            return [xi]
        elif pname in evidence:
            return [evidence[pname]]
        else:
            return self.variables[pname].values


    def _send_messages_to_root(self, node, evidence, children, lambda_msgs):
        """
        Sends λ-messages from the leaves up to the given node in the Bayesian Network.

        This function recursively computes λ-messages for a node by aggregating messages
        from its children, incorporating evidence when available, as part of belief propagation.

        Args:
            node (str): The variable to which λ-messages are being sent.
            evidence (dict): Observed values for some variables.
            children (dict): A mapping from each variable to its list of children.
            lambda_msgs (dict): Stores already computed λ-messages per variable.

        Returns:
            dict: The λ-message for the given node, mapping each of its values to a likelihood.
        """
        # Avoid redundant computation by reusing cached λ-message
        if node in lambda_msgs:
            return lambda_msgs[node]
        var = self.variables[node]
        lambda_msg = {val: 1.0 for val in var.values}
        for child in children[node]:
            child_lambda = self._send_messages_to_root(child, evidence, children, lambda_msgs)
            child_var = self.variables[child]
            parent_names = [p.name for p in child_var.cpt.parents]
            new_lambda = {}
            for xi in var.values:
                msg = 0.0
                for xj in child_var.values:
                    # Collect possible values for each parent, constrained by evidence and current xi
                    all_pa_vals = itertools.product(*[self._get_vals(pname, xi, node, evidence) for pname in parent_names])
                    for pa in all_pa_vals:
                        pa_dict = dict(zip(parent_names, pa))
                        pa_vals = tuple(pa_dict[p.name] for p in child_var.cpt.parents)
                        prob = child_var.cpt.entries[pa_vals][xj]
                        msg += prob * child_lambda[xj]
                new_lambda[xi] = lambda_msg[xi] * msg
            lambda_msg = new_lambda
        # Override λ-message with hard evidence if the variable is observed
        if node in evidence:
            observed = evidence[node]
            lambda_msg = {val: (1.0 if val == observed else 0.0) for val in var.values}
        lambda_msgs[node] = lambda_msg
        return lambda_msg
    

    def _send_messages_from_root(self, node, pi_msg, evidence, children, lambda_msgs, beliefs, pi_msgs):
        """
        Sends π-messages from the root to all its descendants in the Bayesian Network.

        This function computes beliefs for all reachable nodes starting from the root,
        using upward (λ) and downward (π) message passing as defined by Pearl's belief propagation algorithm.

        Args:
            node (str): The current variable from which messages are sent.
            pi_msg (dict): The π-message from the parent to the current node (prior belief).
            evidence (dict): Observed values for some variables.
            children (dict): A mapping from each variable to its children.
            lambda_msgs (dict): Precomputed λ-messages from children to parents.
            beliefs (dict): Stores the resulting marginal beliefs per variable.
            pi_msgs (dict): Stores the π-messages sent to each variable.

        Returns:
            None: Updates `beliefs` and `pi_msgs` in place.
        """
        var = self.variables[node]
        lambda_msg = lambda_msgs[node]
        # Combine π and λ messages to compute the final belief for this node
        belief = {val: pi_msg[val] * lambda_msg[val] for val in var.values}
        beliefs[node] = self._normalize(belief)
        pi_msgs[node] = pi_msg
        for child in children[node]:
            child_var = self.variables[child]
            parent_names = [p.name for p in child_var.cpt.parents]
            # Initialize π-message for each possible value of the child
            child_pi = {xj: 0.0 for xj in child_var.values}
            for xj in child_var.values:
                total = 0.0
                # Determine all parent value combinations, constrained by evidence
                all_pa_vals = itertools.product(*[self._get_vals(pname, None, None, evidence) for pname in parent_names])
                for pa in all_pa_vals:
                    pa_dict = dict(zip(parent_names, pa))
                    pa_vals = tuple(pa_dict[p.name] for p in child_var.cpt.parents)
                    prob = child_var.cpt.entries[pa_vals][xj]
                    # Compute contribution of each parent value using upstream π-messages
                    contrib = 1.0
                    for pname, pval in pa_dict.items():
                        contrib *= self._get_pi_contribution(pname, pval, pi_msgs, evidence)
                    total += contrib * prob
                child_pi[xj] = total
            # Recurse on the child node with the computed π-message
            self._send_messages_from_root(child, child_pi, evidence, children, lambda_msgs, beliefs, pi_msgs)

    def query_marginal(self, query_var, evidence):
        """
        Computes the marginal distribution of a variable given observed evidence using belief propagation.

        Args:
            query_var (str): The name of the variable to query.
            evidence (dict): A dictionary mapping observed variable names to their values.

        Returns:
            dict: A probability distribution over the values of `query_var`, normalized to sum to 1.
        """
        children = self._get_children()
        lambda_msgs = {}
        beliefs = {}
        pi_msgs = {}
        root = next(iter(evidence)) if evidence else query_var
        self._send_messages_to_root(root, evidence, children, lambda_msgs)
        root_var = self.variables[root]
        # If root has parents, no prior is available, so assume uniform prior
        if root_var.cpt.parents:
            root_pi = {val: 1.0 for val in root_var.values}
        else:
            root_pi = root_var.cpt.entries[tuple()]
        self._send_messages_from_root(root, root_pi, evidence, children, lambda_msgs, beliefs, pi_msgs)
        # Fallback in case the target query_var wasn't covered in the first propagation
        if query_var not in beliefs:
            self._send_messages_to_root(query_var, evidence, children, lambda_msgs)
            # Again, use uniform prior if the query_var has parents
            if self.variables[query_var].cpt.parents:
                root_pi = {val: 1.0 for val in self.variables[query_var].values}
            else:
                root_pi = self.variables[query_var].cpt.entries[tuple()]
            self._send_messages_from_root(query_var, root_pi, evidence, children, lambda_msgs, beliefs, pi_msgs)
        return beliefs[query_var]
    
    def query_joint(self, var1, var2, evidence):
        """
        Computes the joint probability distribution of two variables given partial evidence.

        Uses belief propagation to compute:
            P(var1, var2 | evidence) = P(var1 | var2, evidence) * P(var2 | evidence)

        Args:
            var1 (str): The name of the first variable.
            var2 (str): The name of the second variable.
            evidence (dict): Observed values for other variables in the network.

        Returns:
            dict: A nested dictionary {val1: {val2: prob}} representing the normalized joint distribution.
        """
        joint_dist = {}
        for val1 in self.variables[var1].values:
            joint_dist[val1] = {}
            for val2 in self.variables[var2].values:
                extended_evidence = dict(evidence)
                extended_evidence[var1] = val1
                extended_evidence[var2] = val2
                # Fallback in case var2 is not in the marginal or val2 is missing
                marg = self.query_marginal(var2, evidence)
                if var2 not in marg or val2 not in marg:
                    marg = self.query_marginal(var2, evidence)
                if val2 not in marg:
                    continue
                # Fallback in case var1 is not in the conditional or val1 is missing
                cond = self.query_marginal(var1, {**evidence, var2: val2})
                if var1 not in cond or val1 not in cond:
                    cond = self.query_marginal(var1, {**evidence, var2: val2})
                joint_dist[val1][val2] = cond[val1] * marg[val2]
        return self._normalize_nested(joint_dist)

    def _normalize_nested(self, joint_dist):
        """
        Normalizes a nested joint distribution so that all probabilities sum to 1.

        Args:
            joint_dist (dict): Nested dictionary {val1: {val2: prob}}.

        Returns:
            dict: Normalized joint distribution. If total is 0, returns the original.
        """
        total = sum(v for d in joint_dist.values() for v in d.values())
        if total == 0:
            return joint_dist
        for k1 in joint_dist:
            for k2 in joint_dist[k1]:
                joint_dist[k1][k2] /= total
        return joint_dist

# =============================== #
#        MODEL EVALUATION         #
# =============================== #
def _clean_missing_dataframe(df):
    """
    Cleans a dataframe containing missing values and numeric strings.

    Replaces "?" with NaN, and converts all non-missing values to stringified integers.

    Args:
        df (pd.DataFrame): The input dataframe with potential missing values and mixed types.

    Returns:
        pd.DataFrame: The cleaned dataframe with consistent string integer values and NaNs.
    """
    df = df.replace("?", pd.NA)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(int(float(x))) if pd.notna(x) else pd.NA)
    return df



def _evaluate_network(bn, test_df, miss_df):
    """
    Evaluates the accuracy and average confidence of a Bayesian Network on test data with missing values.

    Args:
        bn (BayesianNetwork): The trained Bayesian network.
        test_df (pd.DataFrame): Dataset with complete ground truth values.
        miss_df (pd.DataFrame): Same dataset with some values replaced by NaN.

    Returns:
        float: Accuracy, the proportion of correctly predicted missing values.
        float: Average confidence of the predicted values.
    """
    correct = 0
    total = 0
    confidences = []
    for i in range(len(miss_df)):
        evidence = miss_df.iloc[i].dropna().to_dict()
        ground_truth = test_df.iloc[i].to_dict()
        missing_vars = [col for col in miss_df.columns if pd.isna(miss_df.iloc[i][col])]
        for var in missing_vars:
            predicted_dist = bn.query_marginal(var, evidence)
            if not predicted_dist:
                continue
            predicted_val = max(predicted_dist, key=predicted_dist.get)
            confidence = predicted_dist[predicted_val]
            confidences.append(confidence)
            is_correct = (predicted_val == str(ground_truth[var]))
            correct += is_correct
            total += 1
    accuracy = correct / total if total > 0 else 0.0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return accuracy, avg_confidence
    

def main(train_dataset_file, test_dataset_file, missing_value_file, bayesian_network_file):
    """
    Executes the full pipeline: loading data, training a Bayesian network, and evaluating it.

    Args:
        train_dataset_file (str): Path to the CSV file containing the training data.
        test_dataset_file (str): Path to the CSV file containing the full test data.
        missing_value_file (str): Path to the CSV file containing the test data with missing values.
        bayesian_network_file (str): Path to the Bayesian Network model BIF file to load for evaluation.

    Returns:
        None
    """
    # ── Load ──────────────────────────────
    train_df = pd.read_csv(train_dataset_file)
    test_df = pd.read_csv(test_dataset_file)
    miss_df = _clean_missing_dataframe(pd.read_csv(missing_value_file))
    # ── Train ─────────────────────────────
    bn = BayesianNetwork()
    # TODO: add task3 here 
    # ── Test ──────────────────────────────
    bn = BayesianNetwork(bayesian_network_file)
    acc, _ = _evaluate_network(bn, test_df, miss_df)
    print(f"Accuracy: {acc:.4f}")
    
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python src/inginious.py <train_dataset_file> <test_dataset_file> <missing_value_file> <bayesian_network_file>")
        exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
