import os

import pytorch_lightning as pl
import torch
import torch.ao.quantization as tq
import torchvision.models as models
from on_device_ai.odai.numeric_sensitivity.utils import (
    analyze_acc_combine,
    analyze_accuracy,
    analyze_sensitivity,
    analyze_sensitivity_combine,
    calibrate,
    get_qconfig,
    prepare_sensitivity,
)
from on_device_ai.tools.data.imagenet.loader import build_imagenet_dataloader


# Define constants
train_batch_size = 64
val_batch_size = 64
test_batch_size = 64
num_calib_epochs = 100
num_analyze_epochs = 10
num_acc_epochs = 100
num_acc_combine_epochs = 100
num_combine_epochs = 10
module_list = [
    "module.features.0",
    "module.features.1",
    "module.features.2",
    "module.features.3",
    "module.features.4",
    "module.features.5",
    "module.features.6",
    "module.features.7",
    "module.features.8",
    "module.features.9",
]
pre_trained = True
dataset_name = "cifar10"
model_name = "mobilenet"
model_state_dict_path = ""
weight_bitwidth = 4
activation_bitwidth = 8
weight_qscheme = torch.per_tensor_affine
activation_qscheme = torch.per_tensor_affine
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define dataset
dataset = "imagenet"

# Define model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
model = tq.QuantWrapper(model).to(device)

# Define data module in pytorch lightning
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


data_module = DataModule(os.getcwd(), train_batch_size, val_batch_size, test_batch_size)
data_module.prepare_data()
data_module.setup()


# Define Qconfig and Qconfig Dict
qconfig = get_qconfig(
    weight_bit_width=weight_bitwidth,
    activation_bit_width=activation_bitwidth,
    weight_qscheme=weight_qscheme,
    activation_qscheme=activation_qscheme,
)
qconfig_dict = {}
for mod in module_list:
    qconfig_dict[mod] = qconfig

# Prepare sensitivity
q_model = prepare_sensitivity(model, qconfig_dict)

# Calibrate the prepared model
calibrate(q_model, data_module.train_dataloader(), num_calib_epochs, device)

# Collect L2 errors per module results
results = analyze_sensitivity(
    q_model,
    data_module.test_dataloader(),
    num_analyze_epochs,
    device,
)

# Sort results based on module sensitivity. Prepare sensitivity profile of the model
results_sort = sorted(results.items(), key=lambda x: x[1].item(), reverse=True)

# Get a list of sorted module names
sensitive_sorted = [x[0] for x in results_sort]
sensitive_sorted = sensitive_sorted[1:]  # Skip the full model entry

# Collect L2 errors for top/bottom (3) most sensitive layers
for k in range(2, 4):
    results.update(
        analyze_sensitivity_combine(
            q_model,
            sensitive_sorted,
            k,
            True,
            data_module.test_dataloader(),
            num_combine_epochs,
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
            num_combine_epochs,
            device,
        )
    )


# Collect accuracies per module results
results_acc = analyze_accuracy(q_model, data_module.test_dataloader(), device)

# Collect accuracy results for top/bottom (3) most sensitive layers
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
