import os
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Split Qwen .pt state_dict into per-layer shards")
    parser.add_argument("--input", dest="input_path", default=os.environ.get("QWEN_WEIGHT_PATH", ""))
    parser.add_argument("--output", dest="output_dir", default=os.environ.get("QWEN_WEIGHT_SHARDS_DIR", ""))
    parser.add_argument("--num-layers", dest="num_layers", type=int, default=36)
    parser.add_argument("--model-dir", dest="model_dir", default=os.environ.get("QWEN_MODEL_DIR", ""))
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_dir and not args.input_path:
        args.input_path = os.path.join(args.model_dir, "qwen2.5-3b-instruct.pt")
    if args.model_dir and not args.output_dir:
        args.output_dir = os.path.join(args.model_dir, "weights_shards")

    if not args.input_path or not args.output_dir:
        raise SystemExit("Please set --input/--output or QWEN_WEIGHT_PATH/QWEN_WEIGHT_SHARDS_DIR/QWEN_MODEL_DIR.")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading state_dict: {args.input_path}")
    state_dict = torch.load(args.input_path, map_location="cpu")

    def save_dict(name, d):
        path = os.path.join(args.output_dir, name)
        torch.save(d, path)
        print(f"Saved: {path}")

    # Embedding
    embed_key = "model.embed_tokens.weight"
    if embed_key in state_dict:
        save_dict("embed_tokens.pt", {embed_key: state_dict[embed_key]})
    else:
        raise SystemExit(f"Missing key: {embed_key}")

    # Final norm
    norm_key = "model.norm.weight"
    if norm_key in state_dict:
        save_dict("norm.pt", {norm_key: state_dict[norm_key]})
    else:
        raise SystemExit(f"Missing key: {norm_key}")

    # LM head (optional)
    lm_head_key = "lm_head.weight"
    if lm_head_key in state_dict:
        save_dict("lm_head.pt", {lm_head_key: state_dict[lm_head_key]})

    # Per-layer shards
    for i in range(args.num_layers):
        prefix = f"model.layers.{i}."
        layer_dict = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not layer_dict:
            raise SystemExit(f"No weights found for layer {i} (prefix {prefix})")
        save_dict(f"layer_{i:03d}.pt", layer_dict)

    print("Done.")


if __name__ == "__main__":
    main()
