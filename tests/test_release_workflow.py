"""Regression checks for release paths, data selection and evaluation handoffs."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Codes"))
from experiment_config import physical_fold, target_dir


def load_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def stage_module(stage, filename):
    names = ["model_config", "snr_filter", "encoder", "decoder", "separator", "normalization",
             "train_disjoint", "train_mixed_source_disjoint", "train_mixed_ssl_source_disjoint"]
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    old_path = sys.path[:]
    sys.path.insert(0, str(ROOT / "Codes" / stage))
    try:
        yield load_module(f"Codes/{stage}/{filename}", "release_test_module")
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = old_path


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        for key in list(os.environ):
            if key.startswith(("ESD_JASSNET_", "HLSCMDS_", "EXPH_")):
                del os.environ[key]
        os.environ["ESD_JASSNET_ROOT"] = str(self.root)
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = str(self.root / "matplotlib")

    def config(self, stage="MIXED_TUNING"):
        return runpy.run_path(str(ROOT / "Codes" / stage / "model_config.py"))

    def test_canonical_path_and_legacy_precedence(self):
        canonical = target_dir(self.root, 2)
        self.assertEqual(canonical.name, "hlscmds_full_40x40_10x10_no_unused_fold2")
        legacy = canonical.with_name(canonical.name.replace("hlscmds_", "torabi_"))
        legacy.mkdir(parents=True)
        self.assertEqual(target_dir(self.root, 2), legacy)
        canonical.mkdir()
        self.assertEqual(target_dir(self.root, 2), canonical)
        os.environ["HLSCMDS_DATASET_DIR"] = "legacy_override"
        os.environ["HLSCMDS_TARGET_DIR"] = "explicit_override"
        self.assertEqual(target_dir(self.root, 2), Path("explicit_override"))

    def test_invalid_physical_fold_rejected(self):
        os.environ["HLSCMDS_FOLD"] = "6"
        with self.assertRaises(ValueError):
            physical_fold()

    def test_stage1_produces_stage2_input(self):
        os.environ["ESD_JASSNET_TRAINING_MODE"] = "stage1"
        stage1 = self.config()
        self.assertEqual(stage1["STAGE"], "supervised_from_scratch")
        self.assertFalse(stage1["MIXED_TRAINING"])
        self.assertEqual((stage1["FINETUNE_EPOCHS"], stage1["FINETUNE_LR"], stage1["FINETUNE_PATIENCE"]), (30, 1e-4, 5))
        self.assertEqual(stage1["SUPERVISED_DIR"], stage1["SYNTH_SUPERVISED_DIR"])
        os.environ["ESD_JASSNET_TRAINING_MODE"] = "stage2"
        stage2 = self.config()
        self.assertEqual(Path(stage2["PRETRAIN_CKPT"]), Path(stage1["CKPT_DIR"]) / "scratch_fold1_best.pt")

    def test_target_scratch_keeps_target_dataset(self):
        os.environ.update(HLSCMDS_FOLD="3", ESD_JASSNET_TRAINING_MODE="target_scratch")
        config = self.config()
        self.assertIn("fold3", config["SUPERVISED_DIR"])
        self.assertIn("SCRATCH_FOLD3", config["CKPT_DIR"])
        self.assertFalse(config["MIXED_TRAINING"])
        self.assertEqual(config["FINETUNE_LR"], 1e-4)

    def test_stage3_handoff_and_fold_isolation(self):
        for fold in range(1, 6):
            os.environ["HLSCMDS_FOLD"] = str(fold)
            stage2 = self.config()
            stage3 = self.config("SSL_MIXED")
            self.assertEqual(Path(stage3["PRETRAIN_CKPT"]), Path(stage2["CKPT_DIR"]) / "finetune_fold1_best.pt")
            self.assertEqual(Path(stage3["SSL_PSEUDO_DIR"]).parent.name, f"v1_real_ssl_pseudo_fold{fold}")
            self.assertEqual(stage3["ONLY_FOLD"], 1)

    def test_pseudo_manifest_follows_override(self):
        os.environ["ESD_JASSNET_SSL_PSEUDO_DIR"] = str(self.root / "historical" / "confident")
        self.assertEqual(Path(self.config("SSL_MIXED")["SSL_CONFIDENCE_MANIFEST"]), self.root / "historical" / "manifest_pseudo_confident.csv")

    def test_evaluation_overrides_do_not_change_training(self):
        os.environ["ESD_JASSNET_EVAL_DATA_DIR"] = str(self.root / "external")
        for stage in ["MIXED_TUNING", "SSL_MIXED"]:
            config = self.config(stage)
            self.assertNotEqual(config["EVAL_DATA_DIR"], config["SUPERVISED_DIR"])
            self.assertEqual(Path(config["EVAL_SPLIT_CSV"]), self.root / "external" / "source_disjoint_split_smoke.csv")

    def test_teacher_cli_root_and_explicit_precedence(self):
        module = load_module("Codes/SSL_MIXED/dataset_generation/02_generate_v1_ssl_pseudo_labels.py", "teacher_test")
        os.environ["ESD_JASSNET_STAGE2_CKPT"] = str(self.root / "override.pt")
        other = self.root / "other"
        with patch.object(sys, "argv", ["generator", "--project-root", str(other)]):
            args = module.parse_args()
        self.assertEqual(args.teacher_ckpt, self.root / "override.pt")
        self.assertEqual(args.out_dir.parent, other / "dataset" / "processed")
        with patch.object(sys, "argv", ["generator", "--teacher-experiment", "original_run"]):
            args = module.parse_args()
        self.assertIsNone(args.teacher_ckpt)
        self.assertEqual(args.teacher_experiment, "original_run")

    def test_noise_physical_fold_does_not_change_internal_split(self):
        os.environ["HLSCMDS_FOLD"] = "4"
        module = load_module("Codes/ERROR_ANALYSIS/03_white_noise_robustness_audit.py", "noise_test")
        self.assertEqual(module.PHYSICAL_FOLD, 4)
        self.assertEqual(module.FOLD, 1)
        self.assertIn("fold4", module.DATASET_DIR.name)

    def test_single_checkpoint_uses_only_validation_originals(self):
        import torch
        split = self.root / "split.csv"
        pd.DataFrame({"fold_no": [1, 1], "base_id": [1, 2], "split": ["train", "val"]}).to_csv(split, index=False)
        for stage in ["MIXED_TUNING", "SSL_MIXED"]:
            with self.subTest(stage=stage), stage_module(stage, "evaluate_polarity_control.py") as module:
                dataset = SimpleNamespace(root=self.root, names=["000001_s000_orig", "000002_s000_orig", "000002_s000_aug"])
                args = SimpleNamespace(test_dir=str(self.root), sr=4000, seg_samples=8000,
                                       use_source_disjoint_split=True, source_disjoint_split_csv=str(split),
                                       ckpt="chosen.pt", save_audio=False, save_audio_references=False,
                                       peak_normalise_saved_audio=False, apply_mixture_polarity_calibration=True,
                                       apply_mixture_gain_calibration=True)
                with patch.object(module, "get_triplet_dataset_class", return_value=lambda *a, **kw: dataset), \
                     patch.object(module, "read_target_gain_info_from_checkpoint", return_value={}), \
                     patch.object(module, "load_model"), patch.object(module, "run_inference", return_value=[]) as infer, \
                     contextlib.redirect_stdout(io.StringIO()):
                    module.evaluate_single_ckpt(args, torch.device("cpu"))
                names = [paths[0].name for paths in infer.call_args.kwargs["test_files"]]
                self.assertEqual(names, ["M_000002_s000_orig.wav"])

    def test_evaluator_honors_checkpoint_environment(self):
        os.environ["ESD_JASSNET_EVAL_CKPT"] = "explicit.pt"
        for stage in ["MIXED_TUNING", "SSL_MIXED"]:
            with stage_module(stage, "evaluate_polarity_control.py") as module:
                self.assertEqual(module.Config.ckpt, "explicit.pt")

    def test_architecture_size_and_forward_compatible_across_stages(self):
        import torch
        torch.set_num_threads(1)
        for stage, filename in [("MIXED_TUNING", "train_mixed_source_disjoint.py"),
                                ("SSL_MIXED", "train_mixed_ssl_source_disjoint.py")]:
            with stage_module(stage, filename) as module, contextlib.redirect_stdout(io.StringIO()):
                model = module.build_model(module.Config(), torch.device("cpu")).eval()
                self.assertEqual(sum(p.numel() for p in model.parameters()), 300608)
                with torch.no_grad():
                    heart, lung = model(torch.zeros(1, 1, 8000))
                self.assertEqual(heart.shape, (1, 1, 8000))
                self.assertEqual(lung.shape, heart.shape)
                self.assertTrue(torch.isfinite(heart).all())

    def test_hflung_requires_explicit_hs_input(self):
        module = load_module("Codes/BUILD_HFLUNG_RESPIRATORYTR/build_hflung_selected_20x20_external_val.py", "hflung_test")
        with patch.object(sys, "argv", ["builder"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                module.parse_args()

    def test_hflung_overlap_and_unknown_split_fail_closed(self):
        sys.path.insert(0, str(ROOT / "Codes" / "BUILD_HFLUNG_RESPIRATORYTR"))
        self.addCleanup(sys.path.remove, str(ROOT / "Codes" / "BUILD_HFLUNG_RESPIRATORYTR"))
        module = load_module("Codes/BUILD_HFLUNG_RESPIRATORYTR/build_hflung_selected_20x20_external_val.py", "hflung_overlap_test")
        manifest = self.root / "exph.csv"
        reference = pd.DataFrame({"source_id": ["record_a", "record_b"], "source_path": ["a.wav", "b.wav"], "split": ["train", "val"]})
        reference.to_csv(manifest, index=False)
        args = SimpleNamespace(project_root=self.root, exph_hs_selected_csv=manifest, allow_exph_train_overlap=False)
        with self.assertRaisesRegex(RuntimeError, "overlap EXP_H training"):
            module.check_hs_overlap(args, reference.iloc[:1].copy())
        _, summary = module.check_hs_overlap(args, reference.iloc[1:].copy())
        self.assertEqual(summary["overlap_train_count"], 0)
        self.assertEqual(summary["overlap_val_count"], 1)
        args.allow_exph_train_overlap = True
        _, summary = module.check_hs_overlap(args, reference.iloc[:1].copy())
        self.assertEqual(summary["overlap_train_count"], 1)
        reference.drop(columns="split").to_csv(manifest, index=False)
        with self.assertRaisesRegex(RuntimeError, "identify its train and val"):
            module.check_hs_overlap(args, reference.iloc[:1].copy())

    def test_hflung_relative_audio_paths_resolve_under_root(self):
        module = load_module("Codes/BUILD_HFLUNG_RESPIRATORYTR/build_hflung_selected_20x20_external_val.py", "hflung_paths_test")
        (self.root / "source.wav").touch()
        manifest = self.root / "hs.csv"
        pd.DataFrame({"source_id": ["new_hs"], "source_path": ["source.wav"]}).to_csv(manifest, index=False)
        args = SimpleNamespace(project_root=self.root, hs_selected_csv=manifest, hs_quality_csv=None, hs_split="all", n_hs=1)
        result = module.read_hs_sources(args)
        self.assertEqual(Path(result.iloc[0]["source_path"]), self.root / "source.wav")

    def test_hflung_small_build_exports_audited_additive_validation(self):
        import soundfile as sf
        sys.path.insert(0, str(ROOT / "Codes" / "BUILD_HFLUNG_RESPIRATORYTR"))
        self.addCleanup(sys.path.remove, str(ROOT / "Codes" / "BUILD_HFLUNG_RESPIRATORYTR"))
        module = load_module("Codes/BUILD_HFLUNG_RESPIRATORYTR/build_hflung_selected_20x20_external_val.py", "hflung_build_test")
        t = np.arange(60000) / 4000
        sf.write(self.root / "external_heart.wav", 0.2 * np.sin(2 * np.pi * 60 * t), 4000, subtype="FLOAT")
        sf.write(self.root / "external_lung.wav", 0.1 * np.sin(2 * np.pi * 350 * t), 4000, subtype="FLOAT")
        hs, exph, rank = [self.root / name for name in ["hs.csv", "exph.csv", "rank.csv"]]
        pd.DataFrame({"source_id": ["external_heart"], "source_path": ["external_heart.wav"]}).to_csv(hs, index=False)
        pd.DataFrame({"source_id": ["train_heart"], "source_path": ["train_heart.wav"], "split": ["train"]}).to_csv(exph, index=False)
        pd.DataFrame({"source_name": ["external_lung.wav"], "relative_path": ["external_lung.wav"]}).to_csv(rank, index=False)
        argv = ["builder", "--project-root", str(self.root), "--hs-selected-csv", str(hs),
                "--exph-hs-selected-csv", str(exph), "--hflung-root", str(self.root),
                "--hflung-rank-csv", str(rank), "--n-hs", "1", "--n-ls", "1", "--snrs", "0",
                "--out-name", "hflung_test_1x1"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module.build(module.parse_args())
        processed = self.root / "dataset" / "processed" / "hflung_test_1x1"
        self.assertEqual(len(list(processed.glob("M_*.wav"))), 27)
        split = pd.read_csv(processed / "source_disjoint_split_smoke.csv")
        self.assertEqual(set(split["split"]), {"val"})
        mix = next(processed.glob("M_*.wav"))
        m, _ = sf.read(mix)
        h, _ = sf.read(mix.with_name("H_" + mix.name[2:]))
        l, _ = sf.read(mix.with_name("L_" + mix.name[2:]))
        np.testing.assert_allclose(m, h + l, atol=1e-6)
        selected = self.root / "dataset" / "HFLUNG_TEST_1X1_SELECTED"
        self.assertTrue((selected / "selected_hs_physionet_1.csv").exists())
        self.assertTrue((selected / "hs_overlap_summary.json").exists())

    def test_ssl_confidence_manifest_cannot_silently_fall_back(self):
        (self.root / "M_000001_s000_orig.wav").touch()
        with stage_module("SSL_MIXED", "train_mixed_ssl_source_disjoint.py") as module:
            with self.assertRaises(FileNotFoundError):
                module.SSLPseudoTripletDataset(str(self.root), str(self.root / "missing.csv"))
            manifest = self.root / "manifest.csv"
            pd.DataFrame({"name": ["000002_s000_orig"]}).to_csv(manifest, index=False)
            with self.assertRaisesRegex(RuntimeError, "missing 1 dataset samples"):
                module.SSLPseudoTripletDataset(str(self.root), str(manifest))

    def test_bootstrap_fold_keys_prevent_cross_fold_pairing(self):
        module = load_module("Codes/ERROR_ANALYSIS/02_bootstrap_ci_error_analysis.py", "bootstrap_test")
        output = self.root / "run" / "metrics"
        output.mkdir(parents=True)
        data = pd.DataFrame({"sample_id": ["same", "same"], "base_id": [1, 1], "fold": [1, 1],
                             "physical_fold": [1, 2], "score": [2.0, 8.0]})
        path = output / "per_sample_error_metrics.csv"
        data.to_csv(path, index=False)
        baseline = module.load_run_csv(self.root, "baseline", "run")
        self.assertEqual(baseline["base_id"].nunique(), 2)
        run = baseline.copy()
        run["score"] += 3.0
        result = module.bootstrap_paired_delta_ci(baseline, run, "score", 20, np.random.default_rng(42))
        self.assertEqual(result["n_rows"], 2)
        self.assertEqual(result["n_base"], 2)
        data.drop(columns="physical_fold").to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            module.load_run_csv(self.root, "baseline", "run")


if __name__ == "__main__":
    unittest.main()
