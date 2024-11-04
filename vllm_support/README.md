# OLMo VLLM Release Processes

This folder contains scripts and instructions for running OLMo models using VLLM for efficient text generation.

## Installation

Install the required dependencies:

```bash
pip3 install ai2-olmo vllm torch transformers
```

## Usage

### Converting OLMo Models for VLLM Compatibility

To convert an OLMo checkpoint for use with VLLM:

```bash
python3 vllm_support/olmo_vllm_compatible.py <path_to_checkpoint> <norm_reordering_flag>
```

#### Parameters:
- `path_to_checkpoint`: Path to your OLMo model checkpoint
- `norm_reordering_flag`: Boolean flag for normalization reordering (true/false)

#### Example:
```bash
python3 vllm_support/olmo_vllm_compatible.py model-checkpoints/peteish7/step11931-unsharded-hf/ true
```

## Requirements

- Python 3.12
- CUDA-compatible GPU (recommended)
- Sufficient RAM for model loading