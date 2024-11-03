import argparse
import json
import os
import yaml
from omegaconf import OmegaConf as om
from hf_olmo.convert_olmo_to_hf import convert_checkpoint


def add_auto_map(checkpoint_dir: str):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoConfig": "configuration_olmo.OLMoConfig",
        "AutoModelForCausalLM": "modeling_olmo.OLMoForCausalLM",
        "AutoTokenizer": ["tokenization_olmo_fast.OLMoTokenizerFast", "tokenization_olmo_fast.OLMoTokenizerFast"]
    }
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    try:
        with open(os.path.join(checkpoint_dir, "tokenizer_config.json")) as f:
            config = json.load(f)
        config["auto_map"] = {
            "AutoConfig": "configuration_olmo.OLMoConfig",
            "AutoTokenizer": ["tokenization_olmo_fast.OLMoTokenizerFast", "tokenization_olmo_fast.OLMoTokenizerFast"]
        }
        with open(os.path.join(checkpoint_dir, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    except:
        pass

def fix_bad_tokenizer(checkpoint_dir: str):
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    assert config["tokenizer"]["identifier"] == "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json", "all configs should have bad problem tokenizer before fix"
    config["tokenizer"]['identifier'] = "allenai/eleuther-ai-gpt-neox-20b-pii-special"

    # overwrite config.yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def fix_bad_tokenizer2(checkpoint_dir: str):
    with open(os.path.join(checkpoint_dir, "tokenizer_config.json")) as f:
        config = json.load(f)

    config["eos_token"] = "|||IP_ADDRESS|||"
    config["eos_token_id"] = 50279
    config["padding_token_id"] = 1

    with open(os.path.join(checkpoint_dir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def fix_bad_tokenizer3(checkpoint_dir: str):
    path = os.path.join(checkpoint_dir, "config.yaml")
    conf = om.load(path)
    conf["tokenizer"]["identifier"] = "allenai/gpt-neox-olmo-dolma-v1_5"
    conf["model"]["eos_token_id"] = 50279
    om.save(conf, path)

def main():
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, making it easier to load weights as HF models."
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Location of OLMo checkpoint.",
    )
    
    args = parser.parse_args()

    fix_bad_tokenizer3(args.checkpoint_dir)
    convert_checkpoint(checkpoint_dir=args.checkpoint_dir, ignore_olmo_compatibility=True)
    add_auto_map(args.checkpoint_dir)
    #fix_bad_tokenizer2(args.checkpoint_dir)


if __name__ == "__main__":
    main()
