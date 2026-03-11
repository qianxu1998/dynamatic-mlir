import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SPEC = importlib.util.spec_from_file_location("switching_compare", SCRIPT_DIR / "test.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load switching_testing/test.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from vcd_parser import VcdParser


class SwitchingAggregateReducerTest(unittest.TestCase):
    def test_clock_signal_names_are_explicitly_excluded(self) -> None:
        self.assertTrue(VcdParser.is_clock_signal_name("tb.duv_inst.addi0.clk"))
        self.assertTrue(VcdParser.is_clock_signal_name("tb.duv_inst.clock"))
        self.assertFalse(VcdParser.is_clock_signal_name("tb.duv_inst.addi0_valid"))
        self.assertFalse(VcdParser.is_clock_signal_name("tb.duv_inst.addi0_result[0]"))

    def test_classify_node_macro_family_matches_expected_groups(self) -> None:
        self.assertEqual(MODULE.classify_node_macro_family("addi3"), "compute")
        self.assertEqual(MODULE.classify_node_macro_family("cond_br4"), "control")
        self.assertEqual(MODULE.classify_node_macro_family("tfifo2"), "buffer")
        self.assertEqual(MODULE.classify_node_macro_family("mc_load1"), "memory")
        self.assertEqual(MODULE.classify_node_macro_family("mystery0"), "other")

    def test_evaluate_aggregate_proxy_sums_target_subset_metrics(self) -> None:
        estimator = {
            "addi0": (11, 1, 0),
            "muli2": (7, 4, 1),
            "fork1": (3, 5, 0),
            "cond_br4": (9, 13, 2),
            "tfifo0": (100, 100, 100),
            "mystery0": (50, 60, 70),
        }
        golden = {
            "addi0": (10, 1, 0),
            "muli2": (6, 4, 1),
            "fork1": (3, 4, 0),
            "cond_br4": (9, 10, 2),
            "tfifo0": (100, 100, 100),
            "mystery0": (50, 60, 70),
        }

        results = MODULE.evaluate_aggregate_proxy(
            estimator,
            golden,
            rel_threshold=0.30,
            zero_abs_threshold=5,
        )
        by_label = {result.spec.label: result for result in results}

        compute_data = by_label["sum_toggle_density__macro_family__compute__role__payload_other"]
        self.assertEqual(compute_data.est_total, 18)
        self.assertEqual(compute_data.golden_total, 16)
        self.assertEqual(compute_data.contributing_nodes, 2)
        self.assertEqual(compute_data.missing_vcd_nodes, ())
        self.assertIsNone(compute_data.violation)

        control_valid = by_label["sum_toggle_density__macro_family__control__role__valid"]
        self.assertEqual(control_valid.est_total, 18)
        self.assertEqual(control_valid.golden_total, 14)
        self.assertEqual(control_valid.contributing_nodes, 2)
        self.assertEqual(control_valid.missing_vcd_nodes, ())
        self.assertIsNone(control_valid.violation)

    def test_evaluate_aggregate_proxy_flags_missing_vcd_nodes(self) -> None:
        estimator = {
            "addi0": (11, 1, 0),
            "cond_br4": (9, 13, 2),
        }
        golden = {
            "addi0": (10, 1, 0),
            "cond_br4": (9, None, 2),
        }

        results = MODULE.evaluate_aggregate_proxy(
            estimator,
            golden,
            rel_threshold=0.10,
            zero_abs_threshold=5,
        )
        by_label = {result.spec.label: result for result in results}

        control_valid = by_label["sum_toggle_density__macro_family__control__role__valid"]
        self.assertEqual(control_valid.est_total, 13)
        self.assertEqual(control_valid.golden_total, 0)
        self.assertEqual(control_valid.missing_vcd_nodes, ("cond_br4",))
        self.assertEqual(control_valid.violation, "missing_vcd_nodes=1")


if __name__ == "__main__":
    unittest.main()
