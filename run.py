import argparse
from src.constants import CHOICES
from src.test_model import calculate_ECs, preds_actual_compare

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameter for passing dataset information")
    parser.add_argument("--choices", choices=CHOICES, required=True, help="Enter 'Train, Test or Compare' if you want to Test the model")
    args = parser.parse_args()

    if args.choices == 'Train-Test':
        calculate_ECs()
    elif args.choices == 'Compare':
        preds_actual_compare()

