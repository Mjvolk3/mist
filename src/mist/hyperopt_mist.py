"""hyperopt_mist.py

Hyperopt parameters

"""
import argparse
import copy
import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import ray
import yaml
from ray import tune
from ray.air.config import RunConfig
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from mist import parsing, utils
from mist.data import datasets, featurizers, splitter
from mist.models import mist_model

# from ray.tune.integration.pytorch_lightning import TuneReportCallback


def score_function(config, base_args, orig_dir=""):
    """score_function.

    Args:
        config: All configs passed by hyperoptimizer
        base_args: Base arguments
        orig_dir: ""
    """
    # tunedir = tune.get_trial_dir()
    # Switch s.t. we can use relative data structures
    os.chdir(orig_dir)

    kwargs = copy.deepcopy(base_args)
    kwargs.update(config)
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Split data
    my_splitter = splitter.get_splitter(**kwargs)

    # Get model class
    model_class = mist_model.MistNet
    kwargs["model"] = model_class.__name__
    kwargs["spec_features"] = model_class.spec_features()
    kwargs["mol_features"] = model_class.mol_features()
    kwargs["dataset_type"] = model_class.dataset_type()

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Redefine splitter s.t. this splits three times and remove subsetting
    split_name, (train, val, _test) = my_splitter.get_splits(spectra_mol_pairs)

    for name, _data in zip(["train", "val"], [train, val]):
        logging.info(f"Len of {name}: {len(_data)}")

    train_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=train, featurizer=paired_featurizer, **kwargs
    )
    val_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=val, featurizer=paired_featurizer, **kwargs
    )
    # Create model
    model = model_class(**kwargs)
    if kwargs.get("ckpt_file") is not None:
        model.load_from_ckpt(**kwargs)

    logging.info(f"Starting fold: {split_name}")
    spec_dataloader_module = datasets.SpecDataModule(
        train_dataset, val_dataset, **kwargs
    )
    kwargs["save_dir"] = tune.get_trial_dir()

    # Train the model and return list of dicts of test loss
    model.train_model(
        spec_dataloader_module,
        log_name="",
        log_version=".",
        tune=True,
        **kwargs,
    )


def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parsing.add_base_args(parser)
    parsing.add_dataset_args(parser)
    parsing.add_train_args(parser)
    parsing.add_mist_args(parser)
    parsing.add_hyperopt_args(parser)
    return parser.parse_args()


def get_param_space(trial, kwargs={}):
    """get_param_space.

    Use optuna to define this dynamically

    """
    # Training params
    trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    scheduler = trial.suggest_categorical("scheduler", [True, False])
    if scheduler:
        trial.suggest_float("lr_decay_frac", 0.9, 0.999, log=True)

    trial.suggest_categorical("weight_decay", [1e-6, 1e-7, 0.0])

    # Model params
    trial.suggest_float("spectra_dropout", 0, 0.3, step=0.1)
    trial.suggest_categorical("hidden_size", [128, 256, 512])
    trial.suggest_int("peak_attn_layers", 1, 5)
    trial.suggest_categorical("recycle_form_encoder", [True, False])

    # Data input params
    add_forward = trial.suggest_categorical("add_forward_specs", [True, False])
    if add_forward:
        trial.suggest_float("frac_orig", 0.1, 0.8, step=0.1)

    augment_data = trial.suggest_categorical("augment_data", [True, False])
    if augment_data:
        pass

    # Iterative  loss
    trial.suggest_float("iterative_loss_weight", 0.0, 1, step=0.1)
    trial.suggest_int("refine_layers", 1, 5)

    # Magma loss
    trial.suggest_int("frag_fps_loss_lambda", 0, 15)

    # Worst k loss
    if kwargs.get("loss_fn", "cosine") != "cosine":
        trial.suggest_float("worst_k_weight", 0, 1)


def get_initial_points(kwargs) -> List[Dict]:
    """get_intiial_points.

    Create dictionaries defining initial configurations to test

    """
    param_list = []
    init_base = {
        "learning_rate": 0.0004,
        "scheduler": True,
        "lr_decay_frac": 0.982,
        "weight_decay": 1e-7,
        "spectra_dropout": 0.1,
        "hidden_size": 256,
        "peak_attn_layers": 5,
        "recycle_form_encoder": True,
        "frac_orig": 0.7,
        "add_forward_specs": True,
        "augment_data": True,
        "iterative_loss_weight": 0.5,
        "refine_layers": 4,
        "frag_fps_loss_lambda": 8,
    }

    if kwargs.get("loss_fn", "cosine") != "cosine":
        init_base["worst_k_weight"] = 0.1

    param_list.append(init_base)
    # cp_1 = copy.deepcopy(init_base)
    # cp_1['frag_fps_loss_lambda'] = 8
    # param_list.append(cp_1)

    # cp_2 = copy.deepcopy(init_base)
    # cp_2['scheduler'] = True
    # cp_2['lr_decay_frac'] = 0.97
    # cp_2['learning_rate'] = 6e-4

    # param_list.append(cp_2)
    return param_list


def run_hyperopt():
    """run_hyperopt."""
    args = get_args()
    kwargs = args.__dict__
    ray.init(_temp_dir="/tmp/ray/ray_slg")

    # Fix base_args based upon tune args
    kwargs["gpu"] = args.gpus_per_trial > 0
    max_t = args.max_epochs
    if kwargs["debug"]:
        kwargs["num_h_samples"] = 10
        kwargs["max_epochs"] = 5
    save_dir = kwargs["save_dir"]

    utils.setup_logger(
        save_dir, log_name="hyperopt.log", debug=kwargs.get("debug", False)
    )

    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Define score function
    trainable = tune.with_parameters(
        score_function, base_args=kwargs, orig_dir=Path().resolve()
    )
    yaml_args = yaml.dump(kwargs, indent=2)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    metric = "val_loss"
    # Include cpus and gpus per trial
    trainable = tune.with_resources(
        trainable, {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}
    )
    get_param_space_kwargs = partial(get_param_space, kwargs=kwargs)
    search_algo = OptunaSearch(
        metric=metric,
        mode="min",
        points_to_evaluate=get_initial_points(kwargs),
        space=get_param_space_kwargs,
    )
    search_algo = ConcurrencyLimiter(search_algo, max_concurrent=args.max_concurrent)
    tuner = tune.Tuner(
        trainable,
        # param_space=param_space,
        tune_config=tune.TuneConfig(
            mode="min",
            metric=metric,
            search_alg=search_algo,
            scheduler=ASHAScheduler(
                max_t=24 * 60 * 60,  # max_t,
                time_attr="time_total_s",
                grace_period=kwargs.get("grace_period"),
                reduction_factor=2,
            ),
            num_samples=kwargs.get("num_h_samples"),
        ),
        run_config=RunConfig(name=None, local_dir=args.save_dir),
    )

    if kwargs.get("tune_checkpoint") is not None:
        ckpt = str(Path(kwargs["tune_checkpoint"]).resolve())
        tuner = tuner.restore(
            path=ckpt,
            restart_errored=True,
        )

    results = tuner.fit()
    best_trial = results.get_best_result()
    output = {"score": best_trial.metrics[metric], "config": best_trial.config}
    out_str = yaml.dump(output, indent=2)
    logging.info(out_str)
    with open(Path(save_dir) / "best_trial.yaml", "w") as f:
        f.write(out_str)

    # Output full res table
    results.get_dataframe().to_csv(
        Path(save_dir) / "full_res_tbl.tsv", sep="\t", index=None
    )


if __name__ == "__main__":
    pass
