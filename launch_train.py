#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import random
import string
import sys

import yaml

# try:
#    import fblearner_launch_utils as flu
# except ImportError:
sys.path.append("mobile-vision/common/tools/")
import fblearner_launch_utils as flu

os.environ["PYTHONWARNINGS"] = "ignore"

WORKING_DIR = None
# flu.set_debug_local()
BASE_DIR = "//on_device_ai/odai/numeric_sensitivity"

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, default="", required=True)
parser.add_argument(
    "--my-configs-backup-dir",
    type=str,
    default="manifold://ondevice_ai_tools/tree/projects/supernet/configs/tmp_files",
    required=False,
)
args = parser.parse_args()

flu.DEFAULT_MANIFOLD_LAUNCH_PATH = "ondevice_ai_tools/tree/workflows"
flu._LAUNCH_FROM_MANIFOLD = True

# READ CONFIG-FILE
args.config_file = os.path.join(BASE_DIR.replace("//", ""), args.config_file)
with open(args.config_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
        print(config)
    except yaml.YAMLError as exc:
        print(exc)


TARGET = BASE_DIR + ":get_sensitivity"

#### copy_my_local_cofig_to_manifold ###
manifold_prefix = "manifold://"
assert args.my_configs_backup_dir.startswith(manifold_prefix)
my_configs_backup_dir = args.my_configs_backup_dir[len(manifold_prefix) :]


def get_random_string(length=8):
    # Random string with the combination of lower and upper case
    letters = string.ascii_letters
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


manifold_file_path = "{}/{}.yml".format(my_configs_backup_dir, get_random_string())
os.system(f"manifold put {args.config_file} {manifold_file_path}")

cloud_config_file_path = f"{manifold_prefix}{manifold_file_path}"
print("config_path", cloud_config_file_path)

buck_run_arg = ["-c", "fbcode.enable_gpu_sections=true"]
flu.set_secure_group(config["fblearner_secure_group"])


working_dir = flu.create_manifold_working_path(
    config["exp_name"], flu.DEFAULT_MANIFOLD_LAUNCH_PATH
)

# {node_index}, {num_nodes}, {workflow_run_id} will be rendered automatically
run_args = [
    "--config-file",
    cloud_config_file_path,
    "--cloud",
    "--machine-rank",
    "{node_index}",
    "--num-machines",
    "{num_nodes}",
    "--workflow-run-id",
    "{workflow_run_id}",
    "--dist-url",
    "zeus://{workflow_run_id}",
]


n_gpu_per_node = config["n_gpu_per_node"]
n_cpu_per_node = config["n_cpu_per_node"]
memory = config["memory_per_node"]
gpu_type = config["gpu_type"]


flu.buck_run_dpp(
    config["exp_name"],
    TARGET,
    run_args,
    num_nodes=config["num_nodes"],
    resources=flu.Resources(
        gpu=n_gpu_per_node,
        cpu=n_cpu_per_node,
        memory=memory,
        capabilities=gpu_type,
    ),
    working_dir=working_dir,
    entitlement=config["entitlement"],
    buck_args=buck_run_arg,
)
