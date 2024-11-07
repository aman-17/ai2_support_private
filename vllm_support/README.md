# OLMo VLLM Release Processes

This folder contains scripts and instructions for running OLMo models using VLLM for efficient text generation.

## Requirements

- Python 3.12
- CUDA-compatible GPU (recommended)
- Sufficient RAM for model loading

## Installation

Install the required dependencies:

```bash
pip3 install ai2-olmo vllm torch transformers
```

## Usage

To test backward compitable OLMo with VLLM:

```bash
python3 test_olmo_backward_compatible.py
```