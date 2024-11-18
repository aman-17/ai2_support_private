# HuggingFace Checkpoint Upload Guide

This guide explains how to upload checkpoints to HuggingFace using Beaker.

## Setup

1. Create a Beaker session:
```bash
beaker session create --bare --gpus 0 --budget ai2/oe-training --mount weka://oe-training-default=/data/input
```

2. Clone the repository:
```bash
git clone https://github.com/2015aroras/transformers.git
cd transformers
```

## Configuration

1. Place the required configuration files in the transformers directory:
   - `.yaml` configuration file
   - `.sh` script file

2. Create experiment:
```bash
beaker experiment create config.yaml
```

## Directory Structure
```
transformers/
├── config.yaml
├── script.sh
└── ...
```

## Notes
- Ensure proper access permissions to the Ai2/OE-training budget
- Verify mount path accessibility before running