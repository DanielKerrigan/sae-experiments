import argparse
from pathlib import Path

import torch
from saefarer.analyzing import analyze
from saefarer.model import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_from_disk


def main(root_dir, k):
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
        root_dir / f"saes/roneneldan/TinyStories-1M/sparsity/{k}/sae.pt", DEVICE
    )

    output_dir = root_dir / f"saes/roneneldan/TinyStories-1M/sparsity/{k}/analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(
        (root_dir / "datasets/roneneldan/TinyStories_tokenized_128").as_posix()
    )

    n_analysis_tokens = 10_000_000
    n_analysis_sequences = n_analysis_tokens // sae.cfg.model_sequence_length
    tokens: torch.Tensor = dataset.shuffle()["input_ids"][0:n_analysis_sequences]  # type: ignore
    tokens = tokens.to(DEVICE)

    analyze(
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        feature_indices=list(range(sae.cfg.d_sae)),
        feature_batch_size=256,
        cfg=sae.cfg,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--ksparsity", help="sparsity", type=int)
    parser.add_argument(
        "-p",
        "--path",
        default="../../..",
        help="path to directory containing datasets and models",
    )
    args = parser.parse_args()

    main(args.path, args.ksparsity)
