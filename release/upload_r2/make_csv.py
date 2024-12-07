import os
import csv
import re

def extract_step_number(folder_name):
    match = re.match(r'step(\d+)-unsharded', folder_name)
    if match:
        return int(match.group(1))
    return None

def get_checkpoint_folders(directory):
    checkpoint_folders = []
    try:
        items = os.listdir(directory)
        for item in items:
            if re.match(r'step\d+-unsharded', item):
                checkpoint_folders.append(item)
    except Exception as e:
        print(f"Error in {directory}: {e}")
    return checkpoint_folders

def generate_checkpoint_csv(directories, base_url, output_file):
    rows = []
    for directory in directories:
        checkpoint_folders = get_checkpoint_folders(directory)
        for folder in checkpoint_folders:
            step = extract_step_number(folder)
            if step is not None:
                checkpoint_dir = f"{base_url}/step{step}-unsharded/"
                rows.append({'Step': step, 'Checkpoint Directory': checkpoint_dir})

    rows.sort(key=lambda x: x['Step'])
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Step', 'Checkpoint Directory']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        print(f"Created CSV file: {output_file} with {len(rows)} entries")

directories = [
    '/data/input/amanr/safetens_unshard',
    '/data/input/amanr/safetens_unshard2'
]

if __name__ == '__main__':
    generate_checkpoint_csv(directories, base_url = "https://olmo-checkpoints.org/ai2-llm/peteish7", output_file='checkpoints.csv')