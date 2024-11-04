import os
from typing import List

def get_checkpoint_names() -> List[str]:
    # Fill this.
    # Expected format: ["step1000-tokens4B", "step2000-tokens8B", ...]
    return []

if __name__ == "__main__":
    checkpoints = get_checkpoint_names()

    checkpoints_file = os.path.dirname(os.path.realpath(__file__))

    with open(checkpoints_file, "w") as f:
        for checkpoint in checkpoints:
            f.write(checkpoint + "\n")

