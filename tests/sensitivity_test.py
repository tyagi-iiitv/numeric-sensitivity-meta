# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import torch.ao.quantization as tq
import torchvision.datasets as datasets
import torchvision.models as models
from manifold.clients.python import ManifoldClient

from on_device_ai.odai.numeric_sensitivity.utils import (
    analyze_acc_combine,
    analyze_accuracy,
    analyze_sensitivity,
    analyze_sensitivity_combine,
    generate_vis,
    get_fuse_list,
    prepare_sensitivity,
)
from torch.ao.pruning import fqn_to_module
from torch.utils.data import DataLoader
from torchvision import transforms


class SensitivityTest(unittest.TestCase):
    def test_fuse_modules(self):
        model = tq.QuantWrapper(models.resnet18(pretrained=True))
        model.eval()
        fuse_list = get_fuse_list(model)
        org_fuse_list = [
            ["module.layer1.0.conv1", "module.layer1.0.bn1", "module.layer1.0.relu"],
            ["module.layer1.0.conv2", "module.layer1.0.bn2"],
            ["module.layer1.1.conv1", "module.layer1.1.bn1", "module.layer1.1.relu"],
            ["module.layer1.1.conv2", "module.layer1.1.bn2"],
            ["module.layer2.0.conv1", "module.layer2.0.bn1", "module.layer2.0.relu"],
            ["module.layer2.0.conv2", "module.layer2.0.bn2"],
            ["module.layer2.0.downsample.0", "module.layer2.0.downsample.1"],
            ["module.layer2.1.conv1", "module.layer2.1.bn1", "module.layer2.1.relu"],
            ["module.layer2.1.conv2", "module.layer2.1.bn2"],
            ["module.layer3.0.conv1", "module.layer3.0.bn1", "module.layer3.0.relu"],
            ["module.layer3.0.conv2", "module.layer3.0.bn2"],
            ["module.layer3.0.downsample.0", "module.layer3.0.downsample.1"],
            ["module.layer3.1.conv1", "module.layer3.1.bn1", "module.layer3.1.relu"],
            ["module.layer3.1.conv2", "module.layer3.1.bn2"],
            ["module.layer4.0.conv1", "module.layer4.0.bn1", "module.layer4.0.relu"],
            ["module.layer4.0.conv2", "module.layer4.0.bn2"],
            ["module.layer4.0.downsample.0", "module.layer4.0.downsample.1"],
            ["module.layer4.1.conv1", "module.layer4.1.bn1", "module.layer4.1.relu"],
            ["module.layer4.1.conv2", "module.layer4.1.bn2"],
        ]

        assert fuse_list == org_fuse_list

    def test_prepare_sensitivity(self):
        model = tq.QuantWrapper(models.alexnet())
        qconfig = tq.get_default_qconfig("fbgemm")
        qconfig_dict = {
            "quant": qconfig,
            "module.features.0": qconfig,
            "module.features.1": qconfig,
            "module.features.3": qconfig,
            "module.features.4": qconfig,
            "module.features.6": qconfig,
            "module.features.7": qconfig,
        }
        q_model = prepare_sensitivity(model, qconfig_dict, fuse=True)

        # Checking if the fake quant layer is added to the new model
        self.assertIsNotNone(
            fqn_to_module(q_model, "module.features.0.weight_fake_quant")
        )
        self.assertIsNotNone(
            fqn_to_module(q_model, "module.features.3.weight_fake_quant")
        )
        self.assertIsNotNone(
            fqn_to_module(q_model, "module.features.6.weight_fake_quant")
        )
        self.assertIsNone(fqn_to_module(q_model, "module.features.2.weight_fake_quant"))

        # Checking for activation_post_process observer function post prepare
        observer_dict = {}
        tq.get_observer_dict(q_model, observer_dict)
        self.assertTrue(
            "module.features.0.activation_post_process" in observer_dict.keys(),
            "observer is not recorded in the dict",
        )
        self.assertTrue(
            "module.features.3.activation_post_process" in observer_dict.keys(),
            "observer is not recorded in the dict",
        )
        self.assertTrue(
            "module.features.6.activation_post_process" in observer_dict.keys(),
            "observer is not recorded in the dict",
        )

    def test_analyze_functions(self):
        # Prepare model using prepare_sensitivity
        model = tq.QuantWrapper(models.alexnet(pretrained=True))
        qconfig = tq.get_default_qat_qconfig("qnnpack")
        qconfig_dict = {
            "module.features.0": qconfig,
        }
        q_model = prepare_sensitivity(model, qconfig_dict, fuse=True)

        # Generating sample data for input
        dataset = datasets.FakeData(
            size=10,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            ),
        )
        data_loader = DataLoader(dataset, batch_size=1)

        # Analyze sensitivity
        self.assertRaises(
            AssertionError,
            analyze_sensitivity,
            q_model,
            data_loader,
            0,
            "cpu",
        )

        results = analyze_sensitivity(q_model, data_loader, 1, "cpu")
        assert len(results) == 2
        assert list(results.keys()) == ["full_model", "module.features.0"]

        # Analyze accuracy
        results = analyze_accuracy(q_model, data_loader, "cpu")
        assert len(results) == 1
        assert list(results.keys()) == ["module.features.0"]
        assert len(results["module.features.0"]) == 2

        # Analyze sensitivity combine
        dummy_module_list = [
            "module.features.0",
            "module.features.3",
            "module.features.6",
            "module.features.8",
        ]

        # Assert top (k) is less than the module list
        self.assertRaises(
            AssertionError,
            analyze_sensitivity_combine,
            q_model,
            dummy_module_list,
            5,
            True,
            data_loader,
            1,
            "cpu",
        )

        # Assert num_batches is greater than 0
        self.assertRaises(
            AssertionError,
            analyze_sensitivity_combine,
            q_model,
            dummy_module_list,
            2,
            True,
            data_loader,
            0,
            "cpu",
        )

        # Return results for top (2) modules
        results = analyze_sensitivity_combine(
            q_model, dummy_module_list, 2, True, data_loader, 1, "cpu"
        )
        assert len(results) == 1
        assert list(results.keys()) == ["top_2"]

        # Analyze accuracy combine tests
        # Assert top (k) is less than the module list length
        self.assertRaises(
            AssertionError,
            analyze_acc_combine,
            q_model,
            dummy_module_list,
            5,
            True,
            data_loader,
            "cpu",
        )

        # Assert k is not 0
        self.assertRaises(
            AssertionError,
            analyze_acc_combine,
            q_model,
            dummy_module_list,
            0,
            True,
            data_loader,
            "cpu",
        )

        results = analyze_acc_combine(
            q_model, dummy_module_list, 1, True, data_loader, "cpu"
        )
        assert len(results) == 1
        assert list(results.keys()) == ["top_1"]

    def test_visualizations(self):
        # Sample output from an FBL run
        outputs = {
            "output": (
                {
                    "full_model": 7.231265611946583,
                    "features.1": 5.843551181256771,
                    "features.1.conv": 5.843551181256771,
                    "features.1.conv.0": 4.919299699366093,
                    "features.1.conv.0.0": 4.919299699366093,
                    "features.1.conv.0.1": 0.0,
                    "features.1.conv.0.2": 0.0,
                    "features.1.conv.1": 0.46843000408262014,
                    "features.1.conv.2": 0.0,
                    "features.1.skip_add": 0.0,
                    "features.3": 0.6326284348033369,
                    "features.3.conv": 0.6313776997849345,
                    "features.3.conv.0": 0.023614886653376743,
                    "features.3.conv.0.0": 0.023614886653376743,
                    "features.3.conv.0.1": 0.0,
                    "features.3.conv.0.2": 0.0,
                    "features.3.conv.1": 0.4029371775686741,
                    "features.3.conv.1.0": 0.4029371775686741,
                    "features.3.conv.1.1": 0.0,
                    "features.3.conv.1.2": 0.0,
                    "features.3.conv.2": 0.1540157904382795,
                    "features.3.conv.3": 0.0,
                    "features.3.skip_add": 0.001366970876006235,
                    "features.5": 0.20372457266785204,
                    "features.5.conv": 0.20313728787004948,
                    "features.5.conv.0": 0.009619147756893653,
                    "features.5.conv.0.0": 0.009619147756893653,
                    "features.5.conv.0.1": 0.0,
                    "features.5.conv.0.2": 0.0,
                    "features.5.conv.1": 0.1296474338741973,
                    "features.5.conv.1.0": 0.1296474338741973,
                    "features.5.conv.1.1": 0.0,
                    "features.5.conv.1.2": 0.0,
                    "features.5.conv.2": 0.049709702434483916,
                    "features.5.conv.3": 0.0,
                    "features.5.skip_add": 0.0006967499839447555,
                    "features.7": 0.4198818360455334,
                    "features.7.conv": 0.4198818360455334,
                    "features.7.conv.0": 0.04274588136468083,
                    "features.7.conv.0.0": 0.04274588136468083,
                    "features.7.conv.0.1": 0.0,
                    "features.7.conv.0.2": 0.0,
                    "features.7.conv.1": 0.010986106630298309,
                    "features.7.conv.1.0": 0.010986106630298309,
                    "features.7.conv.1.1": 0.0,
                    "features.7.conv.1.2": 0.0,
                    "features.7.conv.2": 0.3486521153245121,
                    "features.7.conv.3": 0.0,
                    "features.7.skip_add": 0.0,
                    "features.9": 0.0805827581207268,
                    "features.9.conv": 0.08009079087059945,
                    "features.9.conv.0": 0.008143860257405322,
                    "features.9.conv.0.0": 0.008143860257405322,
                    "features.9.conv.0.1": 0.0,
                    "features.9.conv.0.2": 0.0,
                    "features.9.conv.1": 0.04614298476371914,
                    "features.9.conv.1.0": 0.04614298476371914,
                    "features.9.conv.1.1": 0.0,
                    "features.9.conv.1.2": 0.0,
                    "features.9.conv.2": 0.03690246876794845,
                    "features.9.conv.3": 0.0,
                    "features.9.skip_add": 0.0005225208337833465,
                    "top_2": 5.843551181256771,
                    "bottom_2": 0.0,
                    "top_3": 5.843551181256771,
                    "bottom_3": 0.0,
                },
                {
                    "features.1": [34.68000030517578, 57.913997650146484],
                    "features.1.conv": [34.68000030517578, 57.913997650146484],
                    "features.1.conv.0": [41.1359977722168, 65.14199829101562],
                    "features.1.conv.0.0": [41.1359977722168, 65.14199829101562],
                    "features.1.conv.0.1": [71.33799743652344, 90.03199768066406],
                    "features.1.conv.0.2": [71.33799743652344, 90.03199768066406],
                    "features.1.conv.1": [69.4219970703125, 89.01200103759766],
                    "features.1.conv.2": [71.33799743652344, 90.03199768066406],
                    "features.1.skip_add": [71.33799743652344, 90.03199768066406],
                    "features.3": [69.47000122070312, 88.93199920654297],
                    "features.3.conv": [69.4959945678711, 88.93599700927734],
                    "features.3.conv.0": [71.30799865722656, 90.01200103759766],
                    "features.3.conv.0.0": [71.30799865722656, 90.01200103759766],
                    "features.3.conv.0.1": [71.33799743652344, 90.03199768066406],
                    "features.3.conv.0.2": [71.33799743652344, 90.03199768066406],
                    "features.3.conv.1": [70.32799530029297, 89.47200012207031],
                    "features.3.conv.1.0": [70.32799530029297, 89.47200012207031],
                    "features.3.conv.1.1": [71.33799743652344, 90.03199768066406],
                    "features.3.conv.1.2": [71.33799743652344, 90.03199768066406],
                    "features.3.conv.2": [70.552001953125, 89.62199401855469],
                    "features.3.conv.3": [71.33799743652344, 90.03199768066406],
                    "features.3.skip_add": [71.31800079345703, 90.03799438476562],
                    "features.5": [70.48600006103516, 89.46599578857422],
                    "features.5.conv": [70.45600128173828, 89.45600128173828],
                    "features.5.conv.0": [71.26599884033203, 90.03599548339844],
                    "features.5.conv.0.0": [71.26599884033203, 90.03599548339844],
                    "features.5.conv.0.1": [71.33799743652344, 90.03199768066406],
                    "features.5.conv.0.2": [71.33799743652344, 90.03199768066406],
                    "features.5.conv.1": [70.6500015258789, 89.68599700927734],
                    "features.5.conv.1.0": [70.6500015258789, 89.68599700927734],
                    "features.5.conv.1.1": [71.33799743652344, 90.03199768066406],
                    "features.5.conv.1.2": [71.33799743652344, 90.03199768066406],
                    "features.5.conv.2": [71.1240005493164, 89.9280014038086],
                    "features.5.conv.3": [71.33799743652344, 90.03199768066406],
                    "features.5.skip_add": [71.33399963378906, 90.00799560546875],
                    "features.7": [69.88399505615234, 89.36199951171875],
                    "features.7.conv": [69.88399505615234, 89.36199951171875],
                    "features.7.conv.0": [71.17599487304688, 89.93199920654297],
                    "features.7.conv.0.0": [71.17599487304688, 89.93199920654297],
                    "features.7.conv.0.1": [71.33799743652344, 90.03199768066406],
                    "features.7.conv.0.2": [71.33799743652344, 90.03199768066406],
                    "features.7.conv.1": [71.25599670410156, 89.95800018310547],
                    "features.7.conv.1.0": [71.25599670410156, 89.95800018310547],
                    "features.7.conv.1.1": [71.33799743652344, 90.03199768066406],
                    "features.7.conv.1.2": [71.33799743652344, 90.03199768066406],
                    "features.7.conv.2": [70.22200012207031, 89.52799987792969],
                    "features.7.conv.3": [71.33799743652344, 90.03199768066406],
                    "features.7.skip_add": [71.33799743652344, 90.03199768066406],
                    "features.9": [71.22999572753906, 89.9679946899414],
                    "features.9.conv": [71.20600128173828, 89.95999908447266],
                    "features.9.conv.0": [71.3479995727539, 90.03599548339844],
                    "features.9.conv.0.0": [71.3479995727539, 90.03599548339844],
                    "features.9.conv.0.1": [71.33799743652344, 90.03199768066406],
                    "features.9.conv.0.2": [71.33799743652344, 90.03199768066406],
                    "features.9.conv.1": [71.21599578857422, 90.00399780273438],
                    "features.9.conv.1.0": [71.21599578857422, 90.00399780273438],
                    "features.9.conv.1.1": [71.33799743652344, 90.03199768066406],
                    "features.9.conv.1.2": [71.33799743652344, 90.03199768066406],
                    "features.9.conv.2": [71.25199890136719, 89.9219970703125],
                    "features.9.conv.3": [71.33799743652344, 90.03199768066406],
                    "features.9.skip_add": [71.33999633789062, 90.03399658203125],
                    "org_model": [71.33799743652344, 90.03199768066406],
                    "top_2": [34.68000030517578, 57.913997650146484],
                    "bottom_2": [71.33799743652344, 90.03199768066406],
                    "top_3": [34.68000030517578, 57.913997650146484],
                    "bottom_3": [71.33799743652344, 90.03199768066406],
                },
            )
        }

        # Manifold params
        manifold_bucket = "on_device_ai_cv_publicdata"
        manifold_path = "tree/sensitivity_analysis"
        filenames = [
            "layer_wise_sensitivity_test.png",
            "sensitivity_correlation_test.png",
            "accuracy_correlation_test.png",
            "layer_wise_accuracy_correlation_test.png",
        ]
        client = ManifoldClient(manifold_bucket)

        # Remove plots if already exist from previous tests
        for filename in filenames:
            if client.sync_exists(f"{manifold_path}/{filename}"):
                client.sync_rm(f"{manifold_path}/{filename}")

        # Generate new plots
        generate_vis(outputs["output"], suffix="test")

        # Test if the files exist in the directory
        assert client.sync_exists(
            "tree/sensitivity_analysis/layer_wise_sensitivity_test.png"
        )
        assert client.sync_exists(
            "tree/sensitivity_analysis/sensitivity_correlation_test.png"
        )
        assert client.sync_exists(
            "tree/sensitivity_analysis/accuracy_correlation_test.png"
        )
        assert client.sync_exists(
            "tree/sensitivity_analysis/layer_wise_accuracy_correlation_test.png"
        )


if __name__ == "__main__":
    unittest.main()
