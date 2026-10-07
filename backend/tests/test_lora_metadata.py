"""lora_metadata: ヘッダだけ書き換え、テンソルは1バイトも変えないことの検証(一時コピーのみ)。"""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from app.training import lora_metadata as lm


def _write_tiny(path: Path, metadata: dict | None, pad_header: bool = True) -> bytes:
    a = bytes(range(256)) * 4          # 1024 bytes
    b = bytes(reversed(range(256))) * 2  # 512 bytes
    header = {
        "lora_unet.a.lora_down.weight": {"dtype": "F16", "shape": [512], "data_offsets": [0, 1024]},
        "lora_unet.b.lora_up.weight": {"dtype": "F16", "shape": [256], "data_offsets": [1024, 1536]},
    }
    if metadata is not None:
        header = {"__metadata__": metadata, **header}
    blob = json.dumps(header, separators=(",", ":")).encode()
    if pad_header:
        blob += b" " * ((8 - len(blob) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + a + b)
    return a + b


def _tensor_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    return raw[8 + n:]


class LoraMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_keeps_tensors_and_existing_keys(self):
        p = self.dir / "t-000002.safetensors"
        tensors = _write_tiny(p, {"ss_epoch": "2", "ss_learning_rate": "0.0002"})
        changed = lm.rewrite_metadata(p, {"ss_epoch": "99", "lssn_schema": "1", "lssn_trigger_words": '[{"trigger":"日本語"}]'})
        self.assertTrue(changed)
        self.assertEqual(_tensor_bytes(p), tensors)
        meta = lm.read_metadata(p)
        self.assertEqual(meta["ss_epoch"], "2")  # 既存は上書きしない
        self.assertEqual(meta["lssn_schema"], "1")
        self.assertIn("日本語", meta["lssn_trigger_words"])
        header, n, size = lm.read_header(p)
        self.assertEqual(n % 8, 0)
        self.assertEqual(header["lora_unet.b.lora_up.weight"]["data_offsets"], [1024, 1536])
        self.assertFalse(list(self.dir.glob("*.lssn-tmp")))
        # 同じ内容は再書き込みしない(冪等)
        self.assertFalse(lm.rewrite_metadata(p, {"lssn_schema": "1"}))
        # overwrite指定時のみ上書き
        self.assertTrue(lm.rewrite_metadata(p, {"ss_epoch": "3"}, overwrite=True))
        self.assertEqual(lm.read_metadata(p)["ss_epoch"], "3")

    def test_file_without_metadata(self):
        p = self.dir / "none.safetensors"
        tensors = _write_tiny(p, None)
        self.assertTrue(lm.rewrite_metadata(p, {"lssn_schema": "1"}))
        self.assertEqual(_tensor_bytes(p), tensors)
        self.assertEqual(lm.read_metadata(p), {"lssn_schema": "1"})

    def test_incomplete_file_is_not_touched(self):
        p = self.dir / "partial.safetensors"
        _write_tiny(p, {"a": "b"})
        full = p.read_bytes()
        p.write_bytes(full[:-100])  # 書き込み途中を模擬
        with self.assertRaises(ValueError):
            lm.rewrite_metadata(p, {"lssn_schema": "1"})
        self.assertEqual(p.read_bytes(), full[:-100])

    def test_stamp_checkpoint_fills_thin_trainer_metadata(self):
        p = self.dir / "WASABI_ANIMA-000003.safetensors"
        tensors = _write_tiny(p, {"ss_epoch": "3", "ss_steps": "444", "ss_network_dim": "32"})
        ctx = {
            "run_id": 7, "project_id": 1, "project_name": "Wasabi", "started_at_text": "2026-09-21 08:27:04",
            "started_ts": 1789979224.0, "total_epochs": 20, "steps_per_epoch": 148, "snapshot_id": 25,
            "snapshot": {"id": 25, "snapshot_hash": "abc", "item_count": 74},
            "tag_lists": [["wasabi", "1girl"], ["wasabi", "solo"]],
            "triggers": [{"name": "Wasabi", "type": "identity", "trigger": "wasabi", "images": 2}],
            "instances": [{"name": "Swimsuit", "type": "outfit", "trigger": "wasabi_swim", "images": 1}],
            "env": {"python": "3.11.9", "torch": "2.7.0+cu128"},
            "cfg": {"learning_rate": 0.0002, "optimizer": "AdamW8bit", "scheduler": "cosine", "train_batch_size": 1,
                    "resolution": 768, "repeats": 2, "rank": 32, "alpha": 16.0, "epochs": 20, "model_family": "anima",
                    "base_checkpoint_path": "C:/x/anima-base-v1.0.safetensors", "output_name": "WASABI_ANIMA"},
            "resolved": {"gpu_mapping": {"physical_name": "NVIDIA GeForce RTX 3090 Ti", "physical_total_mb": 24564},
                         "total_steps": 2960, "resolved_training_backend": "anima"},
        }
        self.assertTrue(lm.stamp_checkpoint(ctx, p))
        self.assertEqual(_tensor_bytes(p), tensors)
        meta = lm.read_metadata(p)
        self.assertEqual(meta["ss_steps"], "444")  # トレーナー値を維持
        self.assertEqual(meta["ss_learning_rate"], "0.0002")
        self.assertEqual(meta["lssn_trained_started_at"], "2026-09-21T17:27:04+09:00")
        self.assertEqual(meta["lssn_total_steps"], "2960")
        self.assertEqual(meta["lssn_gpu"], "NVIDIA GeForce RTX 3090 Ti")
        self.assertEqual(meta["lssn_torch"], "2.7.0+cu128")
        self.assertEqual(json.loads(meta["lssn_instances"])[0]["trigger"], "wasabi_swim")
        self.assertEqual(json.loads(meta["ss_tag_frequency"])["2_Wasabi"]["wasabi"], 2)
        self.assertTrue(all(isinstance(v, str) for v in meta.values()))
        self.assertFalse(lm.stamp_checkpoint(ctx, p))  # 2回目は何もしない


if __name__ == "__main__":
    unittest.main()
