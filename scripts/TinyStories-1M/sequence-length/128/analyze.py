import argparse
from pathlib import Path

import torch
from saefarer.analyze import analyze_sae
from saefarer.model import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_from_disk


def main(root_dir):
    """Analyze the SAE"""

    root_dir = Path(root_dir)

    DEVICE = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        root_dir / "models/roneneldan/TinyStories-1M"
    )
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(
        root_dir / "models/EleutherAI/gpt-neo-125M"
    )

    sae = SAE.load(
        root_dir / "saes/roneneldan/TinyStories-1M/sequence-length/128/sae.pt", DEVICE
    )

    output_dir = (
        root_dir / "saes/roneneldan/TinyStories-1M/sequence-length/128/analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(
        (root_dir / "datasets/roneneldan/TinyStories_tokenized_128").as_posix()
    )

    n_analysis_tokens = 10_000_000
    n_analysis_sequences = n_analysis_tokens // sae.cfg.model_batch_size_sequences
    tokens: torch.Tensor = dataset.shuffle()["input_ids"][0:n_analysis_sequences]  # type: ignore
    tokens = tokens.to(DEVICE)

    analyze_sae(
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        feature_indices=list(range(sae.cfg.d_sae)),
        cfg=sae.cfg,
        output_dir=output_dir,
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
