import logging
import os
import sys
from multiprocessing import Pool
from typing import Sequence, Tuple

import amrlib
import amrlib.evaluate.smatch_enhanced as se
import smatch
from hydra._internal.utils import is_under_debugger as _is_under_debugger
from omegaconf import OmegaConf

logger = logging.getLogger("src")
_hit_debug, debugging = True, False


def is_under_debugger():
    if os.environ.get("DEBUG_MODE", "").lower() in ("true", "t", "1", "yes", "y"):
        result = True
    else:
        result = _is_under_debugger()
    global _hit_debug, debugging
    if result and _hit_debug:
        logger.warning("Debug mode.")
        _hit_debug = False
        debugging = True
    return result


OmegaConf.register_new_resolver("in_debugger", lambda x, default=None: x if is_under_debugger() else default)


def in_cluster():
    return os.environ.get("SLURM_JOB_ID") is not None


OmegaConf.register_new_resolver("in_cluster", in_cluster)


def get_cluster_id():
    return os.environ.get("SLURM_JOB_ID")


OmegaConf.register_new_resolver("slurm_job_id", get_cluster_id)


def huggingface_path_helper(name, local_path):
    if os.path.exists(local_path):
        return local_path
    return name


OmegaConf.register_new_resolver("hf", huggingface_path_helper)

OmegaConf.register_new_resolver("eval", eval)

OmegaConf.register_new_resolver(
    "if", lambda cond, yes, no: yes if (cond if isinstance(cond, bool) else eval(cond)) else no
)


amrlib_logger = logging.getLogger("amrlib")
amrlib_logger.setLevel(logging.CRITICAL)
amrlib_logger = logging.getLogger("penman")
amrlib_logger.setLevel(logging.CRITICAL)


# patch amrlib. limit processes=8 to avoid OOM issue.
def compute_smatch(test_entries, gold_entries, processes=8):
    pairs = zip(test_entries, gold_entries)
    mum_match = mum_test = mum_gold = 0
    pool = Pool(processes=processes)
    for (n1, n2, n3) in pool.imap_unordered(se.match_pair, pairs):
        mum_match += n1
        mum_test += n2
        mum_gold += n3
    pool.close()
    pool.join()
    precision, recall, f_score = smatch.compute_f(mum_match, mum_test, mum_gold)
    return precision, recall, f_score


se.compute_smatch = compute_smatch


# pytorch lightning patches
# Although this causes some problems (https://github.com/Lightning-AI/lightning/issues/15689),
# I have use this carefully to avoid breaking anything
try:
    import lightning_lite.strategies.launchers.subprocess_script as _scripts
except ImportError:
    import lightning_fabric.strategies.launchers.subprocess_script as _scripts

import pytorch_lightning.strategies.launchers.subprocess_script as _subprocess


def _hydra_subprocess_cmd(local_rank: int) -> Tuple[Sequence[str], str]:
    import __main__  # local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    import hydra
    from hydra.utils import get_original_cwd, to_absolute_path

    # when user is using hydra find the absolute path
    if __main__.__spec__ is None:  # pragma: no-cover
        command = [sys.executable, to_absolute_path(sys.argv[0])]
    else:
        command = [sys.executable, "-m", __main__.__spec__.name]

    command += sys.argv[1:]

    cwd = get_original_cwd()
    os_cwd = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    command += [f"hydra.run.dir='{os_cwd}'", f"hydra.job.name=train_ddp_process_{local_rank}"]
    return command, cwd


_scripts._hydra_subprocess_cmd = _hydra_subprocess_cmd
_subprocess._hydra_subprocess_cmd = _hydra_subprocess_cmd


import warnings

from pytorch_lightning.utilities.warnings import PossibleUserWarning

warnings.filterwarnings("ignore", category=PossibleUserWarning)
