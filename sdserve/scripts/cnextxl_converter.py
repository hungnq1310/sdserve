import argparse
from sdserve.converter.cnext.pt2onnx import ControlNeXtConverter

CONTROLNET_CONFIG  = {
    'in_channels': [128, 128],
    'out_channels': [128, 256],
    'groups': [4, 8],
    'time_embed_dim': 256,
    'final_out_channels': 320,
}

def convert_models(weight_path, output_path, opset, fp16, sd_xl):
    converter = ControlNeXtConverter(
        CONTROLNET_CONFIG,
        weight_path,
        output_path,
        opset,
        fp16,
        sd_xl,
    )
    print(f"Converting model to ONNX format...")
    converter.convert()
    print(f"Model converted successfully and saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--sd_xl", action="store_true", default=False, help="SD XL pipeline")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")
    parser.add_argument(
        "--opset",
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")
    #
    args = parser.parse_args()
    convert_models(args.ckpt_path, args.output_path, args.opset, args.fp16, args.sd_xl)