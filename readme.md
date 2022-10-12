# Instructions for FB users


## Get Started

This doc covers basics in how to train supernets on a devgpu / fblearner.

Multi-gpu training locally may occasionally result in orphaned processes on your devgpu, which you'll sometimes need to kill manually.


## Train a model

All outputs (log files and checkpoints) will be saved to the working directory, which is specified by `models_save_dir` in the config file.


### Train Locally
```shell
buck run @mode/opt on_device_ai/Tools/arnetv2_applications/applications:main_supernet_imagenet -- \
  --config-file ${CONFIG_FILE} # e.g., on_device_ai/Tools/arnetv2_applications/applications/configs/arnet_supernet_baseline.yml
```

Remark:
- Set `resume` in ${CONFIG_FILE} to resume from a previous checkpoint file (to continue the training process); please make sure `last_epoch` in `lr_scheduler` properly initialized based on the current training batch size
- By default, the training will attempt to resume training from the latest checkpoint in `models_save_dir` when training via fblearner
- We provide a number of training configs for different types of supernets in the folder `configs/`, including an example of training quantized supernets - `qat_arnet_supernet_baseline_kd.yml`


### Run on fblearner

```shell
bento console --file on_device_ai/Tools/arnetv2_applications/applications/launch_train.py -- --\
  --config-file ${CONFIG_FILE}
```

### Other


## Evaulation

You can use the following commands to run evolutionary search.

### Evaluate Locally
```shell
buck run @mode/opt on_device_ai/Tools/arnetv2_applications/applications:main_supernet_imagenet -- \
  --config-file ${CONFIG_FILE} # e.g., on_device_ai/Tools/arnetv2_applications/applications/configs/eval_arnet_evo_search.yml
```

Remark:
- Set `resume` in ${CONFIG_FILE} to evaluate a previous checkpoint file
- Modify other settings accordingly for evolutionary search

### Run on fblearner

```shell
bento console --file on_device_ai/Tools/arnetv2_applications/applications/launch_train.py -- --\
  --config-file ${CONFIG_FILE}
```
