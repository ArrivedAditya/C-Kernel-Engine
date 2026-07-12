from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version/v8/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
BUILD_IR = SCRIPTS / "build_ir_v8.py"


def _load_build_ir():
    spec = importlib.util.spec_from_file_location("build_ir_v8", BUILD_IR)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class NemotronStateShapeTests(unittest.TestCase):
    def test_nemotron_h_mamba2_state_shape_uses_state_dim_not_square_head_dim(self) -> None:
        build_ir = _load_build_ir()
        cfg = build_ir._normalize_manifest_config(
            {
                "num_layers": 8,
                "ssm_projection_layout": "mamba2_v_qk_dt",
                "ssm_conv_history_mode": "kernel_width",
                "recurrent_state_layout": "heads_head_dim_state",
                "embed_dim": 4480,
                "ssm_state_size": 128,
                "ssm_group_count": 8,
                "ssm_time_step_rank": 128,
                "ssm_inner_size": 10240,
                "mamba_head_dim": 80,
                "ssm_conv_kernel": 4,
            }
        )

        self.assertEqual(cfg["recurrent_num_heads"], 128)
        self.assertEqual(cfg["recurrent_head_dim"], 80)
        self.assertEqual(cfg["recurrent_state_heads"], 128)
        self.assertEqual(cfg["recurrent_state_rows"], 80)
        self.assertEqual(cfg["recurrent_state_cols"], 128)
        self.assertEqual(build_ir._recurrent_state_stride_bytes(cfg, "ssm"), 128 * 80 * 128 * 4)

        specs = build_ir.build_activation_specs(cfg, mode="decode", context_len=64)
        self.assertEqual(specs["recurrent_ssm_state"]["shape"], "[8, 128, 80, 128]")
        self.assertEqual(specs["recurrent_ssm_state"]["size"], 8 * 128 * 80 * 128 * 4)


if __name__ == "__main__":
    unittest.main()
