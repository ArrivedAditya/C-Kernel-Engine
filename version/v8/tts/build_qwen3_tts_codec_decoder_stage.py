"""Declare one Qwen codec decoder stage using reusable checked operations."""

from __future__ import annotations

from build_qwen3_tts_codec_first_residual_pool_circuit import _snake, _unit
from build_qwen3_tts_codec_rvq_circuit import checkpoint


def append_decoder_stage(graph: dict, *, stage_name: str, decoder_block: int,
                         input_name: str, input_channels: int,
                         input_frames: int, rate: int) -> str:
    if (isinstance(rate, bool) or not isinstance(rate, int) or rate < 1
            or isinstance(input_channels, bool) or not isinstance(input_channels, int)
            or input_channels < 2 or input_channels % 2
            or isinstance(input_frames, bool) or not isinstance(input_frames, int)
            or input_frames < 1):
        raise ValueError("decoder stage requires even channels, frames and positive rate")
    output_channels = input_channels // 2
    output_frames = input_frames * rate
    kernel = 2 * rate
    activation = f"decoder_{stage_name}_snake_beta_output"
    upsample = f"decoder_{stage_name}_upsample_output"
    op = _snake(input_name, activation, input_channels, input_frames,
                f"codec.decoder.decoder.{decoder_block}.block.0")
    graph["activation_buffers"][activation] = {"shape": [input_channels, input_frames]}
    graph["activation_bindings"][activation] = activation
    graph["semantic_checkpoints"]["exports"][activation] = checkpoint(op, "channel_major")
    graph["block_types"]["codec_semantic"]["body"]["ops"].append(op)
    op = {
        "id": upsample, "op": "audio_conv_transpose1d_dense_crop_right_checked",
        "kernel": "audio_conv_transpose1d_dense_crop_right_channel_major_f32_checked",
        "returns_status": True,
        "weight_refs": {
            "weight": f"codec.decoder.decoder.{decoder_block}.block.1.conv.weight",
            "bias": f"codec.decoder.decoder.{decoder_block}.block.1.conv.bias"},
        "graph_slots": {"inputs": {"input": activation},
                        "outputs": {"output": upsample}},
        "params": {"I": input_channels, "O": output_channels,
                   "T": input_frames, "K": kernel, "U": output_frames,
                   "Q": output_channels * output_frames,
                   "call_constants": {
            "cropped_deconv_input_elements": input_channels * input_frames,
            "cropped_deconv_input_stride": input_frames,
            "cropped_deconv_weight_elements": input_channels * output_channels * kernel,
            "cropped_deconv_bias_elements": output_channels,
            "cropped_deconv_output_elements": output_channels * output_frames,
            "cropped_deconv_output_stride": output_frames,
            "cropped_deconv_scratch_elements": output_channels * output_frames,
            "cropped_deconv_input_channels": input_channels,
            "cropped_deconv_output_channels": output_channels,
            "cropped_deconv_input_frames": input_frames,
            "cropped_deconv_kernel_size": kernel,
            "cropped_deconv_stride": rate,
            "cropped_deconv_padding": 0,
            "cropped_deconv_output_padding": 0,
            "cropped_deconv_right_crop": rate,
            "cropped_deconv_output_frames": output_frames}},
    }
    graph["activation_buffers"][upsample] = {"shape": [output_channels, output_frames]}
    graph["activation_bindings"][upsample] = upsample
    graph["semantic_checkpoints"]["exports"][upsample] = checkpoint(op, "channel_major")
    graph["block_types"]["codec_semantic"]["body"]["ops"].append(op)
    previous = upsample
    for unit, dilation in enumerate((1, 3, 9)):
        previous = _unit(graph, unit, dilation, previous,
                         output_channels, output_frames,
                         decoder_block=decoder_block,
                         stem_prefix=f"decoder_{stage_name}_residual")
    return previous
