import argparse
import os
import shutil
from pathlib import Path

import torch
from torch.onnx import export

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from packaging import version

is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")

def convert_to_fp16(
    model_path
):
    '''Converts an ONNX model on disk to FP16'''
    model_dir=os.path.dirname(model_path)
    # Breaking down in steps due to Windows bug in convert_float_to_float16_model_path
    onnx.shape_inference.infer_shapes_path(model_path)
    fp16_model = onnx.load(model_path)
    fp16_model = convert_float_to_float16(
        fp16_model, keep_io_types=True, disable_shape_infer=True
    )
    # clean up existing tensor files
    shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    # save FP16 model
    onnx.save(fp16_model, model_path)

def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    if is_torch_less_than_1_11:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True,
            opset_version=opset,
        )
    else:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )