"""
Download a Hugging Face model and save it to a local path.
Run at Docker build time so the model is baked into the image.
Default: google/flan-t5-small (text-to-text).
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="google/flan-t5-small",
        help="Hugging Face model ID (e.g. google/flan-t5-small)",
    )
    parser.add_argument(
        "--output-dir",
        default="./model",
        help="Directory to save the model",
    )
    args = parser.parse_args()

    if importlib.util.find_spec("torch") is None:
        print(
            "Error: PyTorch is required to download the model weights.\n"
            "Install it with:\n"
            "  pip install torch\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    from transformers import AutoModelForSeq2SeqLM

    output_dir = Path(args.output_dir).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {args.model_id} to {output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    tokenizer.save_pretrained(str(output_dir))
    model.save_pretrained(str(output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
