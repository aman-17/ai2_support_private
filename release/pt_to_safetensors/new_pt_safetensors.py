import os
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from olmo.safetensors_util import state_dict_to_safetensors_file
import torch

def setup_logger(log_dir: Path):
    """Setup logger to both file and console"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'conversion_{timestamp}.log'
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def convert_file(pt_path: Path, logger):
    """Convert a single .pt file to .safetensors format and delete original"""
    if not pt_path.exists():
        logger.warning(f"File not found: {pt_path}")
        return False
        
    safetensors_path = pt_path.with_suffix('.safetensors')
    
    try:
        logger.info(f"Converting {pt_path} to {safetensors_path}")
        state_dict = torch.load(pt_path, weights_only=True)
        state_dict_to_safetensors_file(state_dict, safetensors_path)
        
        pt_path.unlink()
        logger.info(f"Deleted original file: {pt_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {pt_path}: {str(e)}")
        return False

def process_directory(directory: Path, logger):
    """Process model.pt and optim.pt in given directory"""
    success_count = 0
    model_path = directory / "model.pt"
    optim_path = directory / "optim.pt"
    
    if model_path.exists():
        if convert_file(model_path, logger):
            success_count += 1
    
    if optim_path.exists():
        if convert_file(optim_path, logger):
            success_count += 1
            
    return success_count

def main(root_dir: str, log_dir: str):
    root_path = Path(root_dir)
    log_path = Path(log_dir)
    logger = setup_logger(log_path)
    logger.info(f"Starting conversion process in {root_path}")
    unsharded_dirs = list(root_path.glob("**/*unsharded"))

    def get_step_number(path):
        try:
            name = path.name
            if 'step' in name:
                return int(name.split('step')[1].split('-')[0])
            return float('inf')
        except:
            return float('inf')
    
    unsharded_dirs.sort(key=get_step_number)
    
    total_dirs = len(unsharded_dirs)
    if total_dirs == 0:
        logger.warning(f"No directories ending with 'unsharded' found in {root_path}")
        return
    
    logger.info(f"Found {total_dirs} unsharded directories")
    total_files_converted = 0
    
    with tqdm(unsharded_dirs, desc="Processing directories", unit="dir") as pbar:
        for directory in pbar:
            if directory.is_dir():
                logger.info(f"Processing directory: {directory}")
                success_count = process_directory(directory, logger)
                total_files_converted += success_count
                pbar.set_postfix({
                    'converted': total_files_converted,
                    'current_dir': directory.name
                })
    
    logger.info(f"Conversion complete. Total files converted: {total_files_converted}")
    logger.info(f"Log file can be found at: {log_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert all model.pt and optim.pt files to safetensors format in unsharded directories"
    )
    parser.add_argument("root_dir", help="Root directory to search for unsharded directories")
    parser.add_argument(
        "--log-dir", 
        default="conversion_logs",
        help="Directory to store log files (default: conversion_logs)"
    )
    args = parser.parse_args()
    main(args.root_dir, args.log_dir)