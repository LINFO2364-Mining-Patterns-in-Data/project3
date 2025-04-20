import itertools
import re
import sys
from collections import defaultdict


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
        children = defaultdict(list)
        for var in self.variables.values():
            for parent in var.cpt.parents:
                children[parent.name].append(var.name)
        return children

    def _normalize(self, dist):
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items() if total > 0}

    def _get_pi_contribution(self, pname, pval, pi_msgs, evidence):
        if pname in pi_msgs:
            return pi_msgs[pname][pval]
        elif pname in evidence:
            return 1.0 if evidence[pname] == pval else 0.0
        else:
            return 1.0 / len(self.variables[pname].values)

    def _send_messages_to_root(self, node, evidence, children, lambda_msgs):
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
                    def get_vals(pname):
                        if pname == node:
                            return [xi]
                        elif pname in evidence:
                            return [evidence[pname]]
                        else:
                            return self.variables[pname].values

                    all_pa_vals = itertools.product(*[get_vals(pname) for pname in parent_names])

                    for pa in all_pa_vals:
                        pa_dict = dict(zip(parent_names, pa))
                        pa_vals = tuple(pa_dict[p.name] for p in child_var.cpt.parents)
                        prob = child_var.cpt.entries[pa_vals][xj]
                        msg += prob * child_lambda[xj]
                new_lambda[xi] = lambda_msg[xi] * msg
            lambda_msg = new_lambda

        if node in evidence:
            observed = evidence[node]
            lambda_msg = {val: (1.0 if val == observed else 0.0) for val in var.values}

        lambda_msgs[node] = lambda_msg
        return lambda_msg

    def _send_messages_from_root(self, node, pi_msg, evidence, children, lambda_msgs, beliefs, pi_msgs):
        var = self.variables[node]
        lambda_msg = lambda_msgs[node]
        belief = {val: pi_msg[val] * lambda_msg[val] for val in var.values}
        beliefs[node] = self._normalize(belief)
        pi_msgs[node] = pi_msg

        for child in children[node]:
            child_var = self.variables[child]
            parent_names = [p.name for p in child_var.cpt.parents]
            child_pi = {xj: 0.0 for xj in child_var.values}

            for xj in child_var.values:
                total = 0.0
                def get_vals(pname):
                    if pname in evidence:
                        return [evidence[pname]]
                    else:
                        return self.variables[pname].values

                all_pa_vals = itertools.product(*[get_vals(pname) for pname in parent_names])

                for pa in all_pa_vals:
                    pa_dict = dict(zip(parent_names, pa))
                    pa_vals = tuple(pa_dict[p.name] for p in child_var.cpt.parents)
                    prob = child_var.cpt.entries[pa_vals][xj]

                    contrib = 1.0
                    for pname, pval in pa_dict.items():
                        contrib *= self._get_pi_contribution(pname, pval, pi_msgs, evidence)

                    total += contrib * prob

                child_pi[xj] = total

            self._send_messages_from_root(child, child_pi, evidence, children, lambda_msgs, beliefs, pi_msgs)

    def query_marginal(self, query_var, evidence):
        children = self._get_children()
        lambda_msgs = {}
        beliefs = {}
        pi_msgs = {}
        root = next(iter(evidence)) if evidence else query_var
        self._send_messages_to_root(root, evidence, children, lambda_msgs)
        root_var = self.variables[root]
        if root_var.cpt.parents:
            root_pi = {val: 1.0 for val in root_var.values}
        else:
            root_pi = root_var.cpt.entries[tuple()]
        self._send_messages_from_root(root, root_pi, evidence, children, lambda_msgs, beliefs, pi_msgs)
        if query_var not in beliefs:
            self._send_messages_to_root(query_var, evidence, children, lambda_msgs)
            if self.variables[query_var].cpt.parents:
                root_pi = {val: 1.0 for val in self.variables[query_var].values}
            else:
                root_pi = self.variables[query_var].cpt.entries[tuple()]
            self._send_messages_from_root(query_var, root_pi, evidence, children, lambda_msgs, beliefs, pi_msgs)

        return beliefs[query_var]
    
    def query_joint(self, var1, var2, evidence):
        joint_dist = {}
        for val1 in self.variables[var1].values:
            joint_dist[val1] = {}
            for val2 in self.variables[var2].values:
                extended_evidence = dict(evidence)
                extended_evidence[var1] = val1
                extended_evidence[var2] = val2
                marg = self.query_marginal(var2, evidence)
                if var2 not in marg or val2 not in marg:
                    marg = self.query_marginal(var2, evidence)
                if val2 not in marg:
                    continue
                cond = self.query_marginal(var1, {**evidence, var2: val2})
                if var1 not in cond or val1 not in cond:
                    cond = self.query_marginal(var1, {**evidence, var2: val2})
                joint_dist[val1][val2] = cond[val1] * marg[val2]
        return self._normalize_nested(joint_dist)

    def _normalize_nested(self, joint_dist):
        total = sum(v for d in joint_dist.values() for v in d.values())
        if total == 0:
            return joint_dist
        for k1 in joint_dist:
            for k2 in joint_dist[k1]:
                joint_dist[k1][k2] /= total
        return joint_dist

    

def pipeline(train_dataset_file, test_dataset_file, missing_value_file, bayesian_network_file):
    # ── Load ──────────────────────────────
    # ── Task1 ─────────────────────────────
    # create bn._inference()
    # ── Task2 ─────────────────────────────
    # create bn._learn_parameters(dataset)
    # ── Task3 ─────────────────────────────
    # bn = new bn() using _inference and _learn_parameters
    # ── Task4 ─────────────────────────────
    # _evalutate(bn, dataset, missing_value_file)
    pass
    
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python src/inginious.py <train_dataset_file> <test_dataset_file> <missing_value_file> <bayesian_network_file>")
        exit(1)

    pipeline(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
