# step4.py
import sys
from main import BayesianNetwork
from pathlib import Path

def main():
    if len(sys.argv) != 5:
        print("usage: step4.py train.csv test.csv test_missing.csv out.bif")
        sys.exit(1)

    train, test, miss, out_bif = sys.argv[1:5]

    # 1) try provided complete structure (if any)
    dataset_dir = Path(train).parent
    complete_bif = dataset_dir / f"{dataset_dir.name}_complete.bif"
    best_acc = -1; best_bn = None
    if complete_bif.exists():
        bn = BayesianNetwork(complete_bif)
        bn.learn_parameters(train)
        acc, _ = bn.evaluate(test, miss)
        best_acc, best_bn = acc, bn
        print(f"[complete] accuracy = {acc:.3f}")

    # 2) learn structure from scratch
    bn = BayesianNetwork.learn_structure(train)
    acc, _ = bn.evaluate(test, miss)
    print(f"[learned ] accuracy = {acc:.3f}")
    if acc > best_acc:
        best_bn = bn

    best_bn.write_bif(out_bif)
    print("Best network saved to", out_bif)

if __name__ == "__main__":
    main()

# python3 step4.py datasets/sprinkler/train.csv datasets/sprinkler/test.csv datasets/sprinkler/test_missing.csv datasets/sprinkler/outnamefile.bif