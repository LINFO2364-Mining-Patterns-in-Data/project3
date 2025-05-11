import subprocess

# datasets = ["sprinkler", "asia", "sachs", "water", "alarm", "andes"]
datasets = ["sprinkler"]
#datasets = ["sprinkler", "asia",]

for ds in datasets:
    print("=" * 80)
    print(f"Running experiment for dataset: {ds}")
    print("=" * 80)

    cmd = [
        "python3", "main.py",
        f"datasets/{ds}/train.csv",
        f"datasets/{ds}/test.csv",
        f"datasets/{ds}/test_missing.csv",
        f"datasets/{ds}/best_model.bif"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[v] Finished {ds} successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"[x] Error during {ds}: {e}\n")
