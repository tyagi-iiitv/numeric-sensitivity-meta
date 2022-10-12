import os
from typing import Any, Dict

import pytorch_lightning as pl
import torch.distributed.launcher.fb.flow_launch as launch
import torchvision.models as models
from fblearner.flow import api as flow
from on_device_ai.odai.numeric_sensitivity.utils import (
    analyze_acc_combine,
    analyze_accuracy,
    analyze_sensitivity,
    analyze_sensitivity_combine,
    calibrate,
    evaluate,
    get_qconfig,
    prepare_sensitivity,
)
from on_device_ai.tools.data.imagenet.loader import build_imagenet_dataloader


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, train_batch_size, val_batch_size):
        super().__init__()
        self.data_path = data_path
        self.trainloader, self.testloader = None, None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def prepare_data(self):
        self.train_loader, self.val_loader, _ = build_imagenet_dataloader(
            batch_size=self.train_batch_size,
            val_batch_size=self.val_batch_size,
            ddp_sampler=False,
            ddp_val_sampler=False,
        )

        return

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.val_loader


def get_sensitivity(params: Dict[str, Any]):
    device = "cuda"
    print("Input Params Summary...", params, sep=os.linesep)

    assert params["model_name"] in (
        "resnet",
        "mobilenet",
    ), "Model should be one of resnet or mobilenet"

    if params["model_name"] == "mobilenet":
        model = models.quantization.__dict__["mobilenet_v2"](
            pretrained=True, quantize=False
        ).to(device)
    else:
        model = models.quantization.__dict__["resnet18"](
            pretrained=True, quantize=False
        ).to(device)

    print("Model", params["model_name"], "initialized")

    # Use this if the model is not a torchvision quantization model
    # model = tq.QuantWrapper(model).to(device)

    data_module = DataModule(
        os.getcwd(),
        params["train_batch_size"],
        params["val_batch_size"],
    )

    data_module.prepare_data()
    data_module.setup()

    print("Data Module initialized")

    qconfig = get_qconfig(
        weight_bit_width=params["weight_bitwidth"],
        activation_bit_width=params["activation_bitwidth"],
    )

    # Prepare qconfig dict
    qconfig_dict = {}
    for mod in params["module_list"]:
        qconfig_dict[mod] = qconfig

    print("Qconfig initialized")

    # Test sensitivity steps
    q_model = prepare_sensitivity(model, qconfig_dict)
    data_module.prepare_data()
    data_module.setup()

    print("Starting calibration")
    calibrate(
        q_model, data_module.train_dataloader(), params["num_calib_epochs"], device
    )
    print("Calibration complete, now analyzing model")

    results = analyze_sensitivity(
        q_model,
        data_module.test_dataloader(),
        params["num_analyze_epochs"],
        device,
        params["max_level"],
    )

    print("Analyze step complete, sorting results based on l2 errors")

    results_sort = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # Get a list of sorted module names
    sensitive_sorted = [x[0] for x in results_sort]
    sensitive_sorted.remove("full_model")

    # Collect L2 errors for top/bottom (3) most sensitive layers
    print("collecting top/bottom (3) most sensitive layer stats")
    for k in range(2, 4):
        results.update(
            analyze_sensitivity_combine(
                q_model,
                sensitive_sorted,
                k,
                True,
                data_module.test_dataloader(),
                params["num_combine_epochs"],
                device,
            )
        )
        results.update(
            analyze_sensitivity_combine(
                q_model,
                sensitive_sorted,
                k,
                False,
                data_module.test_dataloader(),
                params["num_combine_epochs"],
                device,
            )
        )

    # Collect accuracies per module results
    print("analyzing accuracy")
    results_acc = analyze_accuracy(
        q_model, data_module.test_dataloader(), device, params["max_level"]
    )

    # Acc of non_quantized model
    print("analyzing accuracy of non-quantized model")
    cur_acc1, cur_acc5 = evaluate(model, data_module.test_dataloader(), device)
    results_acc["org_model"] = [
        cur_acc1.avg.item(),
        cur_acc5.avg.item(),
    ]

    # Collect accuracy results for top/bottom (3) most sensitive layers
    print("analyzing accuracy for top/bottom (3) most sensitive layers")
    for k in range(2, 4):
        results_acc.update(
            analyze_acc_combine(
                q_model,
                sensitive_sorted,
                k,
                True,
                data_module.test_dataloader(),
                device,
            )
        )
        results_acc.update(
            analyze_acc_combine(
                q_model,
                sensitive_sorted,
                k,
                False,
                data_module.test_dataloader(),
                device,
            )
        )
    print(results, results_acc)
    return results, results_acc


@flow.registered(owners=["oncall+on_device_ai"])
@flow.typed()
def workflow(params: Dict[str, Any]):
    elastic_parameters = launch.LaunchConfig(
        min_nodes=params["min_nodes"],
        max_nodes=params["max_nodes"],
        nproc_per_node=params["n_gpus"],
        max_restarts=0,
    )
    resource_requirements = flow.ResourceRequirements(
        cpu=params["n_cpus"], gpu=params["n_gpus"], memory=params["memory"]
    )
    ret = launch.elastic_launch(elastic_parameters, get_sensitivity)(
        params, resource_requirements=resource_requirements
    )
    return ret[0]
