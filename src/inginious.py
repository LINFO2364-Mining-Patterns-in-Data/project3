import re
import sys

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
