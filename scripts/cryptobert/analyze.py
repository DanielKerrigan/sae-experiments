import argparse
from pathlib import Path

from saefarer.analyzing import analyze
from saefarer.config import AnalysisConfig
from saefarer.model import SAE
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datasets import load_from_disk


def main(root_dir):
    """Analyze the SAE"""

    root_dir = Path(root_dir)

    cfg = AnalysisConfig(
        device="cuda",
        dataset_column="input_ids",
        attn_mask_column="attention_mask",
        model_batch_size_sequences=32,
        model_sequence_length=128,
        feature_batch_size=64,
        total_analysis_tokens=10_000_000,
        feature_indices=list(range(64)),
        n_example_sequences=10,
        n_context_tokens=5,
    )

    dataset = load_from_disk(
        (root_dir / "datasets/ElKulako/stocktwits-crypto_tokenized/train").as_posix()
    )

    model_name = "ElKulako/cryptobert"

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    sae = SAE.load(
        root_dir / f"saes/{model_name}/sae.pt",
        cfg.device,
    )

    output_path = root_dir / f"saes/{model_name}/analysis.db"

    analyze(
        cfg=cfg,
        model=model,
        dataset=dataset,  # type: ignore
        sae=sae,
        decode_fn=tokenizer.batch_decode,  # type: ignore
        output_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        default="../../..",
        help="path to directory containing datasets and models",
    )
    args = parser.parse_args()

    main(args.path)
