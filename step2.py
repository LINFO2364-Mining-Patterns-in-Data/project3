# step2.py
import sys
from pathlib import Path
from main import BayesianNetwork

def main():
    if len(sys.argv) != 4:
        print("usage: step2.py structure.bif train.csv out_complete.bif")
        sys.exit(1)

    struct_bif, train_csv, out_bif = sys.argv[1:4]
    bn = BayesianNetwork(struct_bif)
    bn.learn_parameters(train_csv, alpha=1.0)
    bn.write_bif(Path(out_bif))
    print("Parameters learned →", out_bif)

if __name__ == "__main__":
    main()

# # python3 step2.py datasets/sprinkler/sprinkler_complete.bif datasets/sprinkler/train.csv datasets/sprinkler/sprinkler_learned_complete.bif