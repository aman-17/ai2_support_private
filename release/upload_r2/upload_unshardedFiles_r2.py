import boto3
import os
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
import mimetypes
from tqdm import tqdm
import csv
import re

def generate_public_url(folder_name, base_url="https://olmo-checkpoints.org/ai2-llm/peteish7"):
    return f"{base_url}/{folder_name}"

def find_unsharded_folders(base_path):
    unsharded_folders = []
    for root, dirs, _ in os.walk(base_path):
        unsharded_dirs = [d for d in dirs if d.endswith('unsharded')]
        for d in unsharded_dirs:
            full_path = os.path.join(root, d)
            unsharded_folders.append(full_path)
    return unsharded_folders

def upload_file(s3_client, bucket_name, file_path, base_path, prefix='ai2-llm/peteish7'):
    try:
        relative_path = os.path.relpath(file_path, base_path)
        s3_key = f"{prefix}/{relative_path}"
        content_type, _ = mimetypes.guess_type(file_path)
        
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
            
        s3_client.upload_file(
            file_path,
            bucket_name,
            s3_key,
            ExtraArgs=extra_args
        )
        return True, s3_key
    except Exception as e:
        return False, f"Error uploading {relative_path}: {str(e)}"

def create_report(folder_info, output_file="r2_upload_report.csv"):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Step', 'Checkpoint Directory'])
        writer.writerows(folder_info)
    print(f"\nReport generated: {output_file}")

def upload_unsharded_folders(account_id, access_key_id, secret_access_key, bucket_name, root_path):
    s3 = boto3.client(
        's3',
        endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(s3={'addressing_style': 'virtual'})
    )
    unsharded_folders = find_unsharded_folders(root_path)
    folder_info = []
    for folder_path in unsharded_folders:
        files_to_upload = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                parent_path = os.path.dirname(folder_path)
                files_to_upload.append((file_path, parent_path))
        
        print(f"\nUploading files from: {folder_path}")
        progress_bar = tqdm(total=len(files_to_upload), desc="Uploading files")
        successful_uploads = 0
        failed_uploads = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {
                executor.submit(
                    upload_file, 
                    s3, 
                    bucket_name, 
                    file_path, 
                    base_path,
                    'ai2-llm/peteish7'
                ): file_path
                for file_path, base_path in files_to_upload
            }
            for future in future_to_file:
                success, result = future.result()
                if success:
                    successful_uploads += 1
                else:
                    failed_uploads.append(result)
                progress_bar.update(1)
        
        progress_bar.close()
        folder_name = os.path.basename(folder_path)
        folder_name_digit = re.search(r'\d+', folder_name).group()
        public_url = generate_public_url(folder_name)
        folder_info.append([
            folder_name_digit,
            public_url
        ])
    create_report(folder_info)

if __name__ == "__main__":
    CONFIG = {
        'account_id': 'a198dc34621661a1a66a02d6eb7c4dc3',
        'access_key_id': 'ef89def20063915f4af13cf05c623710',
        'secret_access_key': '29685c748bd73b31f91cc904885b6294cf598c50164f169cb8e05ab6fbfd4e4d',
        'bucket_name': 'olmo-checkpoints',
        'root_path': '/myfiles/amanr/safetens_unshard'
    }
    
    upload_unsharded_folders(
        CONFIG['account_id'],
        CONFIG['access_key_id'],
        CONFIG['secret_access_key'],
        CONFIG['bucket_name'],
        CONFIG['root_path']
    )