# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy
from datetime import datetime, timedelta
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

import torch

import torch.ao.quantization as tq

import torch.nn as nn
from manifold.clients.python import ManifoldClient
from sklearn import preprocessing
from torch.ao.pruning import fqn_to_module
from tqdm import tqdm

DEFAULT_TTL = timedelta(days=90)


def get_fuse_list(model):
    r"""
    Helper function to generate fusable models list from a given nn.module model.
    Generates a list of module sequences which can be fused.

    Args:
        module: Module for which fusable components are to be searched

    Return:
        List of list, of prepared sequence of modules which can be fused

    """

    visited = set()
    fuse_list = []
    for n, m in model.named_modules():
        # Check for fusable modules inside sequential blocks
        # skip, if the sequential block was a child of another previously visited sequential block
        if type(m) == nn.Sequential and n not in visited:
            visited.add(n)
            prefix = n
            final_list = []
            cur_list = []
            # Inside current sequential block, search for fusable modules in the children
            for n_child, m_child in m.named_modules():
                visited.add(prefix + "." + n_child)
                # A new fusable module may exist starting from current conv or linear module
                if type(m_child) in [nn.Conv2d, nn.Linear]:
                    if len(cur_list) != 1:
                        final_list.append(cur_list)
                    cur_list = [prefix + "." + n_child]
                # Relu and batchnorm following a conv or linear are to be fused
                elif type(m_child) in [nn.ReLU, nn.BatchNorm2d]:
                    cur_list.append(prefix + "." + n_child)
            if len(cur_list) > 1:
                final_list.append(cur_list)
            fuse_list = fuse_list + final_list[1:]
    return fuse_list


def prepare_sensitivity(
    model,
    qconfig_dict,
    fuse=False,
):

    r"""
    Module level quantization preparation of a floating point (nn.module) model.
    This function attaches fake quants at specified modules in the model after fusion.

    qconfig_dict preparation for fusion:
        All the modules which are to be fused must be present in the qconfig_dict.
        For example model.layer1 has [conv -> bn -> relu] operations which will be fused, then
        the qconfig_dict must include all three modules in layer1, see below.

        qconfig_dict = {
            'layer1.conv': qconfig,
            'layer1.bn': qconfig,
            'layer1.relu': qconfig
        }

    Args:
        model: (nn.module) floating point model with attached Quant Stubs are respective modules
        qconfig_dict: quantization config dict mapping individual modules to a qconfig
        fuse: whether to perform a fuse operation on the input model

    Return:
        prepared_model: input model with fake-quantization operators attached at respective modules
    """

    model.eval()
    if fuse:
        fuse_list = get_fuse_list(model)
        q_model = tq.fuse_modules(model, fuse_list, inplace=False)
    else:
        q_model = copy.deepcopy(model)
        q_model.fuse_model()
    tq.propagate_qconfig_(q_model, qconfig_dict)
    q_model.train()
    tq.prepare_qat(q_model, inplace=True)
    q_model.apply(tq.disable_fake_quant)
    q_model.apply(tq.enable_observer)
    return q_model


def get_depth_modules(model, depth):
    r"""
    Get list of modules within a given depth in a model. Returns all modules if depth=-1.
    """
    mod_list = []
    if depth < 0:
        mod_list = [mod_name for mod_name, _ in model.named_modules()]
    else:
        for mod_name, _ in model.named_modules():
            level = mod_name.count(".")
            if level <= depth:
                mod_list.append(mod_name)
    return mod_list


def analyze_sensitivity(model, data_loader, num_batches, device, max_level=-1):
    r"""
    Analyze sensitivity of a calibrated model prepared using prepare_sensitivity function.
    Using sample data, each individual module is quantized and l2 error is returned between the
    output of the quantized model and the original non-quantized model.

    Args:
        model: (nn.module) calibrated model prepared using the prepare_sensitivity function
        data_iterator: Dataloader with attached iterator - iter(dataloader)
        num_batches: Number of batches to use for analysis
        device: type of device to run the experiment on (cuda, cpu)
        max_level: max level/depth of modules to check for sensitivity
    Return:
        results: l2 error dataframe for each module and full model
        This can be seen as a proxy to the sensitivity of the module towards quantization
    """
    assert num_batches > 0, "num_batches should be greather than 0"
    model.apply(tq.disable_observer)
    model.eval()
    results = {}
    loss = nn.MSELoss()
    data_iterator = iter(data_loader)

    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            data_sample, _ = next(data_iterator)
            data_sample = data_sample.to(device)
            y = model(data_sample)
            # Analyzing full model quantization error
            model.apply(tq.enable_fake_quant)
            cur_err = loss(model(data_sample), y).item()
            prev_err = results.get("full_model", [])
            prev_err.append(cur_err)
            results["full_model"] = prev_err
            model.apply(tq.disable_fake_quant)
            mod_list = get_depth_modules(model, max_level)
            for mod_name in mod_list:
                mod = fqn_to_module(model, mod_name)
                if "weight_fake_quant" not in mod_name and getattr(
                    mod, "qconfig", None
                ):
                    mod.apply(tq.enable_fake_quant)
                    cur_err = loss(model(data_sample), y).item()
                    prev_err = results.get(mod_name, [])
                    prev_err.append(cur_err)
                    results[mod_name] = prev_err
                    mod.apply(tq.disable_fake_quant)

    for mods in results:
        results[mods] = np.mean(results[mods])
    return results


def calibrate(model, data_loader, num_batches, device):
    r"""
    Calibration helper function for sensitivity analysis. Runs the data through a given
    model with enabled observers to setup the observer parameters.

    Calibrates the model in-place.

    Args:
        model: Pytorch model (nn.module) typically passed after the prepare_sensitivity function
    """
    model.eval()
    data_it = iter(data_loader)
    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            images, _ = next(data_it)
            images = images.to(device)
            model(images)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output: output tensor from the model
        target: true label tensor
        topk: accuracies to calculate

    Return:
        Acc values for given k values
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, device):
    r"""
    Calculate Acc1 and Acc5 values for the model given a dataset

    Args:
        model: pytorch model (nn.module)
        data_loader: pytorch dataloader for evaluationl of accuracy
        device: type of device to run the experiment on (cuda, cpu)
    Return:
        Acc1 and Acc5 values

    """
    top1 = AverageMeter("Acc@1", ":6.3f")
    top5 = AverageMeter("Acc@5", ":6.3f")
    model = model.to(device)

    with torch.no_grad():
        for images, target in data_loader:
            images = images.to(device)
            target = target.to(device)
            cur_logits = model(images)
            acc1, acc5 = accuracy(cur_logits, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
    return (top1, top5)


def analyze_accuracy(model, data_loader, device, max_level=-1):
    r"""
    Analyze accuracy of a calibrated model prepared using prepare_sensitivity function.
    Using sample data, each individual module is quantized and accuracy is returned on the
    given data_iterator batches.

    Args:
        model: (nn.module) calibrated model prepared using the prepare_sensitivity function
        data_loader: pytorch dataloader for evaluationl of accuracy
        device: type of device to run the experiment on (cuda, cpu)
        max_level: max level/depth of modules to analyze accuracy for
    Return:
        results: Average accuracy value of the model on given data
    """
    model = model.to(device)
    model.apply(tq.disable_observer)
    model.apply(tq.disable_fake_quant)
    model.eval()
    acc = {}
    mod_list = get_depth_modules(model, max_level)
    for mod_name in tqdm(mod_list):
        mod = fqn_to_module(model, mod_name)
        # Check for modules with a qconfig attribute
        if "weight_fake_quant" not in mod_name and getattr(mod, "qconfig", None):
            mod.apply(tq.enable_fake_quant)
            cur_a1, cur_a5 = evaluate(model, data_loader, device)
            acc[mod_name] = [
                cur_a1.avg.item(),
                cur_a5.avg.item(),
            ]
            mod.apply(tq.disable_fake_quant)
    return acc


def analyze_sensitivity_combine(
    model, modules, k, top, data_loader, num_batches, device
):
    r"""
    Analyze sensitivity for a sorted list of modules based on l2 errors.

    Args:
        model: (nn.module) calibrated model prepared using the prepare_sensitivity function
        modules: sorted list (descending) with module names sorted based on decreasing order of l2 errors
            generated from the results of analyze_sensitivity function
        k: top/bottom (k) modules to quantize
        top: boolean to point whether to look for top (k) modules. If false, bottom (k) modules are chosen
        data_loader: Dataloader for analyzing sensitivity
        num_batches: Number of batches to use for analysis
        device: type of device to run the experiment on (cuda, cpu)

    Return:
        results: L2 error of the model with top/bottom (k) modules quantized compared to the non quantized model
    """
    assert num_batches > 0, "num_batches should be greather than 0"
    assert len(modules) >= k, f"Modules list should contain at least {k} elements"
    assert k > 0, "Value of k should be greater than 0"
    model = model.to(device)
    name = "top_" + str(k)
    if top:
        modules = modules[:k]
    else:
        modules = modules[-k:]
        name = "bottom_" + str(k)
    model.apply(tq.disable_observer)
    model.apply(tq.disable_fake_quant)
    model.eval()
    q_model = copy.deepcopy(model)
    results = []
    loss = nn.MSELoss()
    data_iterator = iter(data_loader)

    for mod_name in modules:
        mod = fqn_to_module(q_model, mod_name)
        if mod:
            mod.apply(tq.enable_fake_quant)

    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            data_sample, _ = next(data_iterator)
            data_sample = data_sample.to(device)
            y = model(data_sample)
            cur_err = loss(q_model(data_sample), y).item()
            results.append(cur_err)

    return {name: np.mean(results)}


def analyze_acc_combine(model, modules, k, top, data_loader, device):
    r"""
    Analyze accuracy for a sorted list of modules based on l2 errors.

    Args:
        model: (nn.module) calibrated model prepared using the prepare_sensitivity function
        modules: sorted list (descending) with module names sorted based on decreasing order of l2 errors
            generated from the results of analyze_sensitivity function
        k: top/bottom (k) modules to quantize
        top: boolean to point whether to look for top (k) modules. If false, bottom (k) modules are chosen
        data_loader: pytorch eval set dataloader to analyze accuracy on
        device: type of device to run the experiment on (cuda, cpu)

    Return:
        results: Accuracy of the model with top/bottom (k) modules quantized
    """
    assert len(modules) >= k, f"Modules list should contain at least {k} elements"
    assert k > 0, "Value of k should be greater than 0"
    model = model.to(device)
    name = "top_" + str(k)
    if top:
        modules = modules[:k]
    else:
        modules = modules[-k:]
        name = "bottom_" + str(k)
    model.apply(tq.disable_observer)
    model.apply(tq.disable_fake_quant)
    model.eval()

    for mod_name in modules:
        mod = fqn_to_module(model, mod_name)
        if mod:
            mod.apply(tq.enable_fake_quant)

    acc1, acc5 = evaluate(model, data_loader, device)
    return {name: [acc1.avg.item(), acc5.avg.item()]}


def get_qconfig(
    weight_bit_width=8,
    weight_qscheme=torch.per_channel_affine,
    activation_bit_width=8,
    activation_qscheme=torch.per_tensor_affine,
):
    r"""
    Generates Qconfig for sensitivity analysis given bit widths of weights and activations.
    Use -1 for both input parameters for no quantization
    """
    if weight_bit_width == -1:
        wt_qconfig = torch.nn.Identity
    else:
        if weight_qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            wt_obs = tq.observer.MovingAverageMinMaxObserver.with_args(
                averaging_constant=0.01
            )
        else:
            wt_obs = tq.observer.MovingAveragePerChannelMinMaxObserver.with_args(
                averaging_constant=0.01
            )

        wt_qconfig = tq.fake_quantize.FusedMovingAvgObsFakeQuantize.with_args(
            observer=wt_obs,
            quant_min=-(2 ** (weight_bit_width - 1)),
            quant_max=2 ** (weight_bit_width - 1) - 1,
            dtype=torch.qint8,
            reduce_range=False,
        )

        if activation_bit_width == -1:
            act_qconfig = torch.nn.Identity
        else:
            if activation_qscheme in [
                torch.per_tensor_affine,
                torch.per_tensor_symmetric,
            ]:
                act_obs = tq.observer.MovingAverageMinMaxObserver.with_args(
                    averaging_constant=0.01
                )
            else:
                act_obs = tq.observer.MovingAveragePerChannelMinMaxObserver.with_args(
                    averaging_constant=0.01
                )

            act_qconfig = tq.fake_quantize.FusedMovingAvgObsFakeQuantize.with_args(
                observer=act_obs,
                quant_min=0,
                quant_max=2 ** (activation_bit_width) - 1,
                dtype=torch.quint8,
                reduce_range=False,
            )

        qconfig = torch.quantization.QConfig(act_qconfig, wt_qconfig)
        return qconfig


def put_manifold_image(
    img,
    filename,
    manifold_bucket="on_device_ai_cv_publicdata",
    manifold_path="tree/sensitivity_analysis",
    ttl=DEFAULT_TTL,
):
    r"""
    Saves a given image buffer to manifold bucket

    Args:
        img: image buffer to be saved
        ttl: timedelta struct specifying number of days to hold the data
    """
    manifold_filename = f"{manifold_path}/{filename}"
    client = ManifoldClient(manifold_bucket)
    img.seek(0)
    if not client.sync_exists(manifold_path):
        client.sync_mkdir(manifold_path, userData=False, ttl=ttl)
    client.sync_put(
        manifold_filename,
        img,
        predicate=ManifoldClient.Predicates.AllowOverwrite,
        userData=False,
        ttl=ttl,
    )
    print(f"Saved {manifold_filename} under {manifold_bucket}")


def generate_vis(
    sensitivity_results,
    suffix=None,
    manifold_bucket="on_device_ai_cv_publicdata",
    manifold_path="tree/sensitivity_analysis",
    ttl=DEFAULT_TTL,
):
    r"""
    Generates and saves visualizations from sensitivity analysis fbl outputs to manifold

    Args:
        sensitivity_results: Dictionary results from sesitivity analysis fbl runs
        suffix: string to append to the filenames before saving
        ttl: timedelta struct specifying number of days to hold the data
    """
    results = sensitivity_results[0]
    results_acc = sensitivity_results[1]

    # Get predicted errors for modules
    pred_err = {}
    for mod_name in results.keys():
        last_mod_pos = mod_name.rfind(".")
        if last_mod_pos != -1:
            prefix, _ = mod_name[:last_mod_pos], mod_name[last_mod_pos + 1 :]
            cur_err = pred_err.get(prefix, 0)
            cur_err += results[mod_name]
            pred_err[prefix] = cur_err

    mod_names_vis = [key for key in pred_err.keys() if key in results.keys()]
    err_pred = [pred_err[key] for key in mod_names_vis]
    err = [results[key] for key in mod_names_vis]

    # Define manifold params
    img_buf = BytesIO()
    if not suffix:
        suffix = str(datetime.now()).replace(" ", "")

    # Generate per module sensitivity scatterplot
    plt.subplots(figsize=(15, 8))
    plt.scatter(mod_names_vis, err, color="k", s=100, label="L2 Error")
    plt.scatter(mod_names_vis, err_pred, color="g", s=100, label="Predicted L2 Error")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc=3, fontsize=10)
    plt.savefig(img_buf, bbox_inches="tight", format="png")
    put_manifold_image(
        img_buf,
        f"layer_wise_sensitivity_{suffix}.png",
        manifold_bucket,
        manifold_path,
        ttl,
    )

    # Generate Correlation plot for predicted and actual l2 errors
    plt.subplots(figsize=(15, 8))
    plt.scatter(err, err_pred, color="g", s=60)
    plt.xlabel("Actual L2 error")
    plt.ylabel("Predicted L2 error")
    plt.savefig(img_buf, bbox_inches="tight", format="png")
    put_manifold_image(
        img_buf,
        f"sensitivity_correlation_{suffix}.png",
        manifold_bucket,
        manifold_path,
        ttl,
    )

    # Generate Accuracy Correlation scatterplot
    acc1_drops = np.array(
        [(results_acc["org_model"][0] - results_acc[key][0]) for key in mod_names_vis]
    ).reshape(-1, 1)
    err = np.array([results[key] for key in mod_names_vis]).reshape(-1, 1)

    plt.subplots(figsize=(15, 8))
    plt.scatter(err, acc1_drops, color="g", s=60)
    plt.xlabel("Accuracy Drop Percentage")
    plt.ylabel("L2 Error")
    plt.savefig(img_buf, bbox_inches="tight", format="png")
    put_manifold_image(
        img_buf,
        f"accuracy_correlation_{suffix}.png",
        manifold_bucket,
        manifold_path,
        ttl,
    )

    # Normalize the accuracy and error values
    scaler = preprocessing.MinMaxScaler()
    acc1_drops_normalize = scaler.fit_transform(acc1_drops)
    err_normalize = scaler.fit_transform(err)

    # Generate per module accuracy drop and sensitivity chart
    plt.subplots(figsize=(15, 8))
    plt.scatter(
        mod_names_vis, acc1_drops_normalize, color="k", s=100, label="Acc1 Drop"
    )
    plt.scatter(mod_names_vis, err_normalize, color="r", s=100, label="Sensitivity")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc=3, fontsize=10)
    plt.savefig(img_buf, bbox_inches="tight", format="png")
    put_manifold_image(
        img_buf,
        f"layer_wise_accuracy_correlation_{suffix}.png",
        manifold_bucket,
        manifold_path,
        ttl,
    )
