import argparse
from pathlib import Path

from saefarer.config import TrainingConfig
from saefarer.training import train
from transformers import AutoModelForCausalLM

from datasets import load_from_disk


def main(root_dir, k):
    """Train the SAE"""

    root_dir = Path(root_dir)

    cfg = TrainingConfig(
        device="cuda",
        dtype="float32",
        # dataset
        dataset_column="input_ids",
        # dimensions
        d_in=64,
        expansion_factor=4,
        # loss functions
        k=k,
        aux_k=32,
        aux_k_coef=1 / 32,
        dead_tokens_threshold=10_000_000,
        hidden_state_index=7,
        normalize=False,
        # batch sizes
        model_sequence_length=128,
        model_batch_size_sequences=32,
        n_batches_in_store=64,
        sae_batch_size_tokens=4096,
        # adam
        lr=3e-4,
        beta1=0.9,
        beta2=0.999,
        eps=6.25e-10,
        # training
        total_training_tokens=100_000_000,
        # logging
        logger="wandb",
        log_batch_freq=500,
        wandb_project="saefarer",
        wandb_group="TinyStories-1M_k",
        wandb_name=f"{k}",
        wandb_notes="Comparing sparsity.",
    )

    model = AutoModelForCausalLM.from_pretrained(
        root_dir / "models/roneneldan/TinyStories-1M"
    )
    model.to(cfg.device)

    dataset = load_from_disk(
        (root_dir / "datasets/roneneldan/TinyStories_tokenized_128").as_posix()
    )

    output_dir = root_dir / f"saes/roneneldan/TinyStories-1M/sparsity/{k}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train(
        cfg=cfg,
        model=model,
        dataset=dataset,  # type: ignore
        save_path=output_dir / "sae.pt",
        log_path=output_dir,
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
