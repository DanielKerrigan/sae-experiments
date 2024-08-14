import argparse
from pathlib import Path

from saefarer.analyzing import analyze
from saefarer.config import AnalysisConfig
from saefarer.model import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_from_disk


def main(root_dir, seq_len):
    """Analyze the SAE"""

    root_dir = Path(root_dir)

    cfg = AnalysisConfig(
        device="cuda",
        dataset_column="input_ids",
        model_batch_size_sequences=32,
        model_sequence_length=seq_len,
        feature_batch_size=256,
        total_analysis_tokens=10_000_000,
        feature_indices=[],
        n_example_sequences=10,
        n_context_tokens=5,
    )

    model = AutoModelForCausalLM.from_pretrained(
        root_dir / "models/roneneldan/TinyStories-1M"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        root_dir / "models/EleutherAI/gpt-neo-125M"
    )

    sae = SAE.load(
        root_dir / f"saes/roneneldan/TinyStories-1M/sequence-length/{seq_len}/sae.pt",
        cfg.device,
    )

    output_path = (
        root_dir
        / f"saes/roneneldan/TinyStories-1M/sequence-length/{seq_len}/analysis.db"
    )

    dataset = load_from_disk(
        (root_dir / f"datasets/roneneldan/TinyStories_tokenized_{seq_len}").as_posix()
    )

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
    parser.add_argument("-l", "--length", help="sequence length", type=int)
    parser.add_argument(
        "-p",
        "--path",
        default="../../..",
        help="path to directory containing datasets and models",
    )
    args = parser.parse_args()

    main(args.path, args.length)
