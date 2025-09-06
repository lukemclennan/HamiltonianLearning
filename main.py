import os
import argparse
import yaml
import importlib
import torch
import numpy as np
import random

seed = 42

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

from src.utils.registers import build_from_config


def set_by_path(cfg, keypath, value):
    """Given cfg dict, a dotted keypath like 'optimizer.args.lr', set cfg[...] = value."""
    keys = keypath.split('.')
    d = cfg
    for k in keys[:-1]:
        if k not in d:
            print(d, k)
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value

def load_group_config(config_dir, group_name, filename):
    path = os.path.join(config_dir, group_name, filename)
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
    
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="./configs/default.yaml", help="Path to the main config file")
    p.add_argument(
        "-O", "--override",
        dest="override",            # ← name it “overrides” in args
        action="append",
        default=[],                  # ← so args.overrides always exists
        help="Override a config key, e.g. optimizer.args.lr=0.005"
    )
    args = p.parse_args()
    config_path = args.config
    base = os.path.dirname(config_path)
    cfg = yaml.safe_load(open(config_path))

    # Parse overrides: convert types
    raw_overrides = []
    for ov in args.override:
        if '=' not in ov: continue
        key, val = ov.split('=', 1)
        if val.lower() in ('true', 'false'):
            v = val.lower() == 'true'
        else:
            try:
                v = int(val)
            except ValueError:
                try:
                    v = float(val)
                except ValueError:
                    v = val
        raw_overrides.append((key, v))

    # Split top-level vs group-level overrides
    top, rest = [], []
    for key, val in raw_overrides:
        if '.' not in key and key in cfg:
            top.append((key, val))
        else:
            rest.append((key, val))
    # Apply top-level overrides
    for key, val in top:
        cfg[key] = val


    # load each subgroup
    dataset_cfg   = load_group_config(base, "dataset",   cfg["dataset"])
    model_cfg     = load_group_config(base, "model",     cfg["model"])
    optimizer_cfg = load_group_config(base, "optimizer", cfg["optimizer"])
    trainer_cfg   = load_group_config(base, "trainer",   cfg["trainer"])

    # Assign default save_dir
    # 1) Peel off the basename (without “.yaml”) for dataset & trainer
    dataset_name = os.path.splitext(cfg["dataset"])[0]   # e.g. “windy_pendulum”
    trainer_name = os.path.splitext(cfg["trainer"])[0]   # e.g. “gda”

    # 2) Build “othermeta” string however you like:
    #    e.g. combine model & optimizer names, plus a timestamp
    model_name     = os.path.splitext(cfg["model"])[0]      # e.g. “ssgp”
    optimizer_name = os.path.splitext(cfg["optimizer"])[0]  # e.g. “adam”
    #timestamp      = datetime.now().strftime("%Y%m%d-%H%M%S")
    other_meta     = f"{model_name}_{optimizer_name}"

    # 3) Compose the final save_dir
    save_dir = os.path.join(
        "checkpoints", 
        f"{dataset_name}_{trainer_name}_{other_meta}"
    )
    # override it for now
    trainer_cfg["args"]["save_dir"] = save_dir


    groups = {
        "dataset" : dataset_cfg,
        "model": model_cfg,
        "optimizer": optimizer_cfg,
        "trainer": trainer_cfg
    }

    # Hard coded CLI override 
    print(rest)
    for key, val in rest:
        if '.' in key:
            grp, sub = key.split('.', 1)
            if grp in groups:
                set_by_path(groups[grp], sub, val)
            else:
                set_by_path(cfg, key, val)
        else:
            if key in groups:
                cfg[key] = val
            else:
                set_by_path(cfg, key, val)


    data_root = cfg["data_root"]
    train_path = dataset_cfg["train_data_file"].format(data_root=data_root)
    test_path  = dataset_cfg["test_data_file"].format(data_root=data_root)

    dataset_cfg = groups["dataset"]
    model_cfg = groups["model"]
    optimizer_cfg = groups["optimizer"]
    trainer_cfg = groups["trainer"]

    #print(groups)
    


    #os.makedirs(save_dir, exist_ok=True)

    if cfg["mode"] == "train":
    # build objects
        train_ds = build_from_config({
                "module": dataset_cfg["module"],
                "class":  dataset_cfg["class"],
                "args":   {**dataset_cfg["args"], "data_path": train_path, "batch_time": dataset_cfg["args"]["batch_time"]}
            })
        val_ds = build_from_config({
            "module": dataset_cfg["module"],
            "class":  dataset_cfg["class"],
            "args":   {**dataset_cfg["args"], "data_path": test_path,  "batch_time": dataset_cfg["args"]["batch_time"]}
        })
        model    = build_from_config(model_cfg)
        optimizer= build_from_config({
            "module": optimizer_cfg["module"],
            "class":  optimizer_cfg["class"],
            "args":   {**optimizer_cfg.get("args", {}), "params": model.parameters()}
        })

        trainer_cls = getattr(importlib.import_module(trainer_cfg["module"]),
                            trainer_cfg["class"])
        print(trainer_cfg.get("args", {}))
        trainer = trainer_cls(
            model=model,
            dataset=train_ds,
            val_dataset=val_ds,
            optimizer=optimizer,
            ckpt_interval = 100,
            **trainer_cfg.get("args", {})
        )
        trainer.train()

    elif cfg["mode"] == "visualize":
        # call your visualize_results pipeline
        from visualize_results import visualize
        model_cfg_path = os.path.join(base, "model", cfg["model"])
        visualize(
             checkpoint_path=cfg["checkpoint"],
             data_path=test_path,
             model_config_path=model_cfg_path,
             output_dir=cfg.get("visualize_args", {}).get("output_dir", "visualizations"),
             num_examples=cfg.get("visualize_args", {}).get("num_examples", 5)
         )

    else:
        raise ValueError(f"Unknown mode {cfg['mode']}")

if __name__=="__main__":
    main()
