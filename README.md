# *LEADS*: Learning Dynamical Systems that Generalize Across Environments (NeurIPS 2021) ([arXiv](https://arxiv.org/abs/2106.04546), [HAL](https://hal.archives-ouvertes.fr/hal-03261055), [OpenReview](https://openreview.net/forum?id=HD6CxZtbmIx))

This repository is the official implementation of *LEADS*: Learning Dynamical Systems that Generalize Across Environments.

## Requirements

We recommend Python 3.7 or greater for installation. You can install all the required by executing the following command

```bash
pip3 install --user -U -r requirements.txt
```

## Training and evaluation code

Use the following command to run the experiments for LV, GS and NS datasets,

```bash
python3 train_leads.py {lv,gs,ns}
```

The experiments are saved into `./exp` by default. You can specify your own experiment directory with `--path /path/to/exp`.

The *LEADS* model is launched by default. If you need to run the experiments for main baselines, you can specify the experiment type with `--decomp_type {leads, leads_no_min, one_for_all, one_per_env}`

CPU is used by default, if you have CUDA compatible device, you can also specify with the option `--device 'cuda:[cuda_device_id]'`

Run `python3 train_leads.py --help` to show the complete list of options.

Evaluation code is integrated into the training process as we need to see how the test performance evolve during training. All outputs are saved in a log file for each experiment.

## Datasets

All datasets can be created on-the-fly. Nonetheless, we provide a buffered version for NS to save your time. They are available in `exp` directory. You can move them to your own directory specified with `--path` to reuse these buffers.

## Results

Our model achieves the following performance on:

| Dataset            | Test MSE        |
| ------------------ |---------------- |
| LV - 10 envs - 1 trajectory/env   |    1.16+-0.99 e-3        |
| GS - 3 envs - 1 trajectory/env | 2.08+-2.88 e-3|
| NS - 4 envs - 8 trajectories/env | 5.95+-3.65 e-3|
