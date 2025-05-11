# step1.py
import sys, json
from main import BayesianNetwork

def main():
    if len(sys.argv) != 3:
        print("usage: step1.py network.bif '{\"Var1\":\"0\", \"Var2\":\"1\"}'")
        sys.exit(1)

    net = BayesianNetwork(sys.argv[1])
    evidence = json.loads(sys.argv[2])

    # infer *all* variables not present in the evidence
    targets  = [v for v in net.vars if v not in evidence]
    post = net.posterior(evidence, targets)

    print("Posterior:", post)

if __name__ == "__main__":
    main()


# posterior on sprinkler network
# python3 step1.py datasets/sprinkler/sprinkler_complete.bif '{"Cloudy":"1","Sprinkler":"0"}'
# python3 step1.py datasets/sprinkler/sprinkler_complete.bif '{"Sprinkler":"1","Cloudy":"1","Rain":"0"}'    
# python3 step1.py datasets/sprinkler/sprinkler_complete.bif '{"Sprinkler":"1"}' 
