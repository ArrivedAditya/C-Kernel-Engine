import importlib.util
import ctypes
import json
import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "resolve_layout_chain_v8.py"
SPEC = importlib.util.spec_from_file_location("resolve_layout_chain_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resolver
SPEC.loader.exec_module(resolver)

BUILD_IR_SCRIPT = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
sys.path.insert(0, str(BUILD_IR_SCRIPT.parent))
BUILD_IR_SPEC = importlib.util.spec_from_file_location("build_ir_v8_layout_test", BUILD_IR_SCRIPT)
assert BUILD_IR_SPEC is not None and BUILD_IR_SPEC.loader is not None
build_ir = importlib.util.module_from_spec(BUILD_IR_SPEC)
BUILD_IR_SPEC.loader.exec_module(build_ir)


def provider(provider_id, role, layout, priority, placement="local"):
    field = "outputs" if role == "producer" else "inputs"
    port = "y" if role == "producer" else "x"
    return {
        "id": provider_id,
        "selection": {"priority": priority},
        field: [{"name": port, "layout": layout, "placement": placement}],
    }


class LayoutProviderSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="cke-layout-provider-")
        cls._lib_path = Path(cls._tmp.name) / "liblayout.so"
        subprocess.run(
            [
                "gcc", "-std=c11", "-O2", "-shared", "-fPIC",
                str(ROOT / "src" / "kernels" / "layout_kernels.c"),
                "-o", str(cls._lib_path),
            ],
            check=True,
        )
        cls._lib = ctypes.CDLL(str(cls._lib_path))
        signature = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        cls._lib.ck_layout_token_to_head_f32.argtypes = signature
        cls._lib.ck_layout_head_to_token_f32.argtypes = signature

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_layout_providers_are_bit_exact_in_both_directions(self):
        tokens, heads, dim = 7, 3, 5
        source = array("f", (float(index) / 7.0 for index in range(tokens * heads * dim)))
        head = array("f", [0.0]) * len(source)
        restored = array("f", [0.0]) * len(source)
        src_ptr = (ctypes.c_float * len(source)).from_buffer(source)
        head_ptr = (ctypes.c_float * len(head)).from_buffer(head)
        restored_ptr = (ctypes.c_float * len(restored)).from_buffer(restored)
        self._lib.ck_layout_token_to_head_f32(
            src_ptr, head_ptr, tokens, heads, dim
        )
        self._lib.ck_layout_head_to_token_f32(
            head_ptr, restored_ptr, heads, tokens, dim
        )
        self.assertEqual(source.tobytes(), restored.tobytes())

    def test_checked_in_layout_converters_match_physical_schema(self):
        schema = json.loads(
            (ROOT / "version" / "v8" / "schemas" / "kernel_physical_layout.schema.json").read_text()
        )
        validator = Draft202012Validator(schema)
        maps = ROOT / "version" / "v8" / "kernel_maps"
        checked = 0
        for path in maps.glob("*.json"):
            document = json.loads(path.read_text())
            if not isinstance(document.get("layout_conversion"), dict):
                continue
            checked += 1
            conversion = document["layout_conversion"]
            for key in ("from_layout", "to_layout"):
                errors = list(validator.iter_errors({"layout": conversion[key]}))
                self.assertEqual(errors, [], path.name)
            self.assertGreater(conversion.get("cost_rank", 0), 0)
        self.assertEqual(checked, 2)

    def test_call_ir_preserves_selected_physical_provider(self):
        lowered = {
            "config": {},
            "operations": [{
                "idx": 7,
                "kernel": "layout_convert_token_to_head_f32",
                "function": "transpose_inplace",
                "op": "transpose_qkv_to_head_major",
                "layer": 0,
                "section": "body",
            }],
        }
        call_ir = build_ir.generate_ir_lower_3(lowered, "prefill")
        operation = call_ir["operations"][0]
        self.assertEqual(operation["kernel"], "layout_convert_token_to_head_f32")
        self.assertEqual(
            operation["resolved_physical_execution"]["layout_conversion"]["to_layout"],
            "head_major_contiguous",
        )

    def test_direct_compatible_chain_beats_higher_priority_converted_chain(self):
        routes = resolver.rank_layout_routes(
            [
                provider("token_fast", "producer", "token_major_contiguous", 500),
                provider("head_direct", "producer", "head_major_contiguous", 100),
            ],
            producer_port="y",
            consumers=[provider("attention", "consumer", "head_major_contiguous", 100)],
            consumer_port="x",
            converters=[{
                "id": "token_to_head",
                "from_layout": "token_major_contiguous",
                "to_layout": "head_major_contiguous",
                "cost_rank": 10,
            }],
        )
        self.assertEqual(routes[0].producer.provider_id, "head_direct")
        self.assertIsNone(routes[0].converter_id)

    def test_priority_ranks_equally_compatible_direct_providers(self):
        routes = resolver.rank_layout_routes(
            [
                provider("baseline", "producer", "head_major_contiguous", 100),
                provider("measured", "producer", "head_major_contiguous", 200),
            ],
            producer_port="y",
            consumers=[provider("attention", "consumer", "head_major_contiguous", 100)],
            consumer_port="x",
        )
        self.assertEqual(routes[0].producer.provider_id, "measured")

    def test_distributed_placement_requires_explicit_transport_converter(self):
        with self.assertRaisesRegex(RuntimeError, "no compatible physical provider chain"):
            resolver.rank_layout_routes(
                [provider("local_q", "producer", "head_major_contiguous", 100, "local")],
                producer_port="y",
                consumers=[provider("remote_attn", "consumer", "head_major_contiguous", 100, "sharded")],
                consumer_port="x",
            )

        routes = resolver.rank_layout_routes(
            [provider("local_q", "producer", "head_major_contiguous", 100, "local")],
            producer_port="y",
            consumers=[provider("remote_attn", "consumer", "head_major_contiguous", 100, "sharded")],
            consumer_port="x",
            converters=[{
                "id": "head_all_to_all",
                "from_layout": "head_major_contiguous",
                "to_layout": "head_major_contiguous",
                "from_placement": "local",
                "to_placement": "sharded",
                "cost_rank": 50,
            }],
        )
        self.assertEqual(routes[0].converter_id, "head_all_to_all")

    def test_missing_layout_fails_closed(self):
        missing = provider("bad", "producer", "head_major_contiguous", 100)
        del missing["outputs"][0]["layout"]
        with self.assertRaisesRegex(RuntimeError, "has no physical layout"):
            resolver.rank_layout_routes(
                [missing],
                producer_port="y",
                consumers=[provider("attention", "consumer", "head_major_contiguous", 100)],
                consumer_port="x",
            )

    def test_equal_rank_chains_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "ambiguous equal-rank"):
            resolver.rank_layout_routes(
                [
                    provider("a", "producer", "head_major_contiguous", 100),
                    provider("b", "producer", "head_major_contiguous", 100),
                ],
                producer_port="y",
                consumers=[provider("attention", "consumer", "head_major_contiguous", 100)],
                consumer_port="x",
            )


if __name__ == "__main__":
    unittest.main()
