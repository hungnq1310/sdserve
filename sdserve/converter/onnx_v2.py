import argparse
from pathlib import Path
import torch
import onnx_graphsurgeon as gs
from onnx import shape_inference
from packaging import version
from polygraphy.backend.onnx.loader import fold_constants
from torch.onnx import export


is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")
is_torch_2_0_1 = version.parse(version.parse(torch.__version__).base_version) == version.parse("2.0.1")


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


class OnnxConverter:

    def get_inputs_names(self, model):
        from inspect import signature
 
        forward_parameters = signature(model.forward).parameters
        forward_inputs_set = list(forward_parameters.keys()) #! Ordered
        return forward_inputs_set

    def optimize(onnx_graph, name, verbose):
        opt = Optimizer(onnx_graph, verbose=verbose)
        opt.info(name + ": original")
        opt.cleanup()
        opt.info(name + ": cleanup")
        opt.fold_constants()
        opt.info(name + ": fold constants")
        # opt.infer_shapes()
        # opt.info(name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(name + ": finished")
        return onnx_opt_graph


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
        with torch.inference_mode(), torch.autocast("cuda"):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_xl", action="store_true", default=False, help="SD XL pipeline")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument(
        "--controlnet_path",
        nargs="+",
        required=True,
        help="Path to the `controlnet` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--opset",
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

    args = parser.parse_args()

    if args.sd_xl:
        from .pipelines.sdxl import StableDiffusionXLConverter
        converter = StableDiffusionXLConverter(args.model_path, args.output_path, args.opset, args.fp16)
        converter.convert_models()
    else:
        from .pipelines.sd15 import StableDiffusionConverter
        converter = StableDiffusionConverter(args.model_path, args.output_path, args.opset, args.fp16)
        converter.convert_models()