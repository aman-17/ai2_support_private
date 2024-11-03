import argparse
import json
import os
from omegaconf import OmegaConf as om
from hf_olmo.convert_olmo_to_hf import write_tokenizer


def add_auto_map(checkpoint_dir: str):
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

def update_yaml_config(checkpoint_dir: str):
    path = os.path.join(checkpoint_dir, "config.yaml")
    conf = om.load(path)
    conf["tokenizer"]["identifier"] = "allenai/gpt-neox-olmo-dolma-v1_5"
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
    update_yaml_config(checkpoint_dir=args.checkpoint_dir)
    write_tokenizer(checkpoint_dir=args.checkpoint_dir)
    os.remove(os.path.join(args.checkpoint_dir, "config.yaml"))
    add_auto_map(args.checkpoint_dir)


if __name__ == "__main__":
    main()
