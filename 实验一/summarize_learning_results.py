import csv
from pathlib import Path

def main():
    project_dir = Path(__file__).resolve().parent
    result_files = [
        "learning_bow_results.csv",
        "learning_embedding_results.csv",
        "learning_bert_results.csv",
    ]
    print(f"{'Method':<24} {'Test Accuracy':>15} {'Test Macro-F1':>15}")
    for filename in result_files:
        with (project_dir / filename).open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                print(
                    f"{row['method']:<24} "
                    f"{float(row['test_accuracy']):>14.2%} "
                    f"{float(row['test_macro_f1']):>14.2%}"
                )

if __name__ == "__main__":
    main()
