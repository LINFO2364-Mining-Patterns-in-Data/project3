# step3.py
import sys
from pathlib import Path
from main import BayesianNetwork

def main():
    if len(sys.argv) != 3:
        print("usage: step3.py train.csv out_learned.bif")
        sys.exit(1)

    train_csv, out_bif = sys.argv[1:3]
    bn = BayesianNetwork.learn_structure(train_csv)
    bn.write_bif(Path(out_bif))
    print("Structure + parameters learned →", out_bif)

if __name__ == "__main__":
    main()

# python3 step3.py datasets/sprinkler/train.csv datasets/sprinkler/out.bif