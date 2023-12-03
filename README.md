# CHAP AMR Parser

This is the official implementation of [AMR Parsing with Causal Hierarchical Attention and Pointers](https://openreview.net/forum?id=SI2CXa5eok), presented at EMNLP 2023.

## Installation

```
chmod +x install.sh
. ./install.sh
```

The installation script will create a new conda environment, named as `chap_amr`, and compile some c++ extension, developed based on [Transformer Grammars](https://github.com/google-deepmind/transformer_grammars).


## Prepare data

First, prepare AMR graph banks using the script in `amrlib`: `https://github.com/bjascob/amrlib/blob/master/scripts/33_Model_Parse_XFM/10_Collect_AMR_Data.py`. You will obtain a directory `tdata_xfm` in `amrlib`'s working directory.
Move it to `<project root>/data/AMR3.0/tdata_xfm` (change to `AMR2.0` if processing the AMR 2.0 graph bank; and you don't need this step when processing OOD data).

Then, run `scripts/10_prepare_data.sh` at the project root. It contains commands of processing all datasets. So you may just need some of them according to your usage.


## Train

```
python src/train.py experiment=<model config>

# our best model
python src/train.py experiment=t2g_point2tgt_paware_strict_adapter
```

We use `hydra` to configure training and models. This project is based on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template). `experiment` will use the config files in `configs/experiment`, in which import the corresponding config files in `configs/model`. The config file name is organized as follows:

| Model Config (configs/model/)        | Coref Layer | Encode Pointer | Tree Modeling |
| ------------------------------------ | :---------: | :------------: | :-----------: |
| t2g_baseline                         |      -      |       -        |       -       |
| t2g_inline_adapter                   |      -      |       -        | CHA (adapter) |
| t2g_inline_inplace                   |      -      |       -        | CHA (inplace) |
| t2g_point2tgt                        |     Yes     |       -        |       -       |
| t2g_point2tgt_paware                 |     Yes     |      Yes       |       -       |
| t2g_point2tgt_paware_strict*         |     Yes     |      Yes       |       -       |
| t2g_point2tgt_paware_strict_adapter* |     Yes     |      Yes       | CHA (adapter) |

* strict means that the pointers of coref layer are constrained to valid values.


## Evaluate

In the training script, a prediction on the test set will be stored under the working directory. To evaluate on a trained model on other datasets, use the following command:
```
python src/eval.py experiment=<model config> data=<data config> ckpt_path=<ckpt path>
```

Then, you can use the scripts in `amrlib` to call `BLINK` to generate AMR graphs with wiki tags and compute the fine-grained metrics. The scripts could be found at `https://github.com/bjascob/amrlib/blob/master/scripts/33_Model_Parse_XFM/`.


## License and Citing

This code is released under the MIT license. 

If you use this work for research, please cite our paper:
```
@inproceedings{
anonymous2023amr,
title={{AMR} Parsing with Causal Hierarchical Attention and Pointers},
author={Anonymous},
booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
year={2023},
url={https://openreview.net/forum?id=SI2CXa5eok}
}

```

