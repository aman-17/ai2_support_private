from google.cloud import storage
from google.oauth2 import service_account
import csv
import re

def create_storage_client(service_account_path):
    """Create authenticated storage client using service account."""
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path
    )
    return storage.Client(credentials=credentials)

def extract_step_number(folder_name):
    """Extract the step number from the folder name."""
    match = re.match(r'step(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None

def get_checkpoint_folders(client, bucket_name, prefix=''):
    """Get all checkpoint folders from a GCS bucket path."""
    checkpoint_folders = set()
    try:
        bucket = client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        for blob in blobs:
            path = blob.name
            if prefix:
                path = path[len(prefix):].lstrip('/')
            
            folder = path.split('/')[0]
            if re.match(r'step\d+', folder):
                checkpoint_folders.add(folder)
                
    except Exception as e:
        print(f"Error in bucket {bucket_name}, prefix {prefix}: {e}")
    
    return list(checkpoint_folders)

def parse_gcs_path(gcs_path):
    """Parse GCS path into bucket name and prefix."""
    # Remove 'gs://' if present
    path = gcs_path.replace('gs://', '')
    
    # For path 'ai2-llm/checkpoints/OLMo-medium/peteish13-highlr'
    # bucket should be 'ai2-llm'
    # prefix should be 'checkpoints/OLMo-medium/peteish13-highlr'
    parts = path.split('/', 1)
    bucket = parts[0]  # 'ai2-llm'
    prefix = parts[1] if len(parts) > 1 else ''  # everything after first '/'
    return bucket, prefix

def generate_checkpoint_csv(service_account_path, gcs_paths, output_file):
    """Generate a CSV file with checkpoint information from GCS buckets."""
    base_url = "https://olmo-checkpoints.org/ai2-llm/peteish13"
    client = create_storage_client(service_account_path)
    rows = []
    
    for gcs_path in gcs_paths:
        bucket_name, prefix = parse_gcs_path(gcs_path)
        print(f"Processing bucket: {bucket_name}, prefix: {prefix}")
        checkpoint_folders = get_checkpoint_folders(client, bucket_name, prefix)
        print(f"Found {len(checkpoint_folders)} checkpoint folders")
        
        for folder in checkpoint_folders:
            step = extract_step_number(folder)
            if step is not None:
                checkpoint_dir = f"{base_url}/step{step}/"
                rows.append({'Step': step, 'Checkpoint Directory': checkpoint_dir})
    
    rows.sort(key=lambda x: x['Step'])
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Step', 'Checkpoint Directory']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        print(f"Created CSV file: {output_file} with {len(rows)} entries")

if __name__ == '__main__':
    # Path to your service account JSON file
    service_account_path = "/data/input/amanr/service_account.json"
    
    gcs_paths = [
        'ai2-llm/checkpoints/OLMo-medium/peteish13-highlr',
        'ai2-llm/checkpoints/OLMo-medium/peteish13-highlr-zlossfix'
    ]
    
    generate_checkpoint_csv(service_account_path, gcs_paths, 'checkpoints13b.csv')
    
    with open('checkpoints13b.csv', 'r') as f:
        print("\nFirst few entries in the CSV:")
        for i, line in enumerate(f):
            if i <= 5:
                print(line.strip())