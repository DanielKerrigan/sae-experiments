import argparse
from pathlib import Path

from saefarer.config import TrainingConfig
from saefarer.training import train
from transformers import AutoModelForSequenceClassification

from datasets import load_from_disk


def main(root_dir):
    """Train the SAE"""

    root_dir = Path(root_dir)

    dataset = load_from_disk(
        (root_dir / "datasets/ElKulako/stocktwits-crypto_tokenized/train").as_posix()
    )

    cfg = TrainingConfig(
        device="cuda",
        dtype="float32",
        # dataset
        dataset_column="input_ids",
        attn_mask_column="attention_mask",
        # dimensions
        d_in=768,
        expansion_factor=4,
        # loss functions
        k=4,
        aux_k=512,
        aux_k_coef=1 / 32,
        dead_tokens_threshold=10_000_000,
        hidden_state_index=10,
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
        total_training_tokens=dataset.shape[0] * 128,  # 136M
        # logging
        logger="wandb",
        log_batch_freq=500,
        wandb_project="saefarer",
        wandb_group="cryptobert",
        wandb_name="Initial",
        wandb_notes="Initial SAE training for cryptobert.",
        # checkpointing
        checkpoint_batch_freq=10_000,
    )

    model_name = "ElKulako/cryptobert"

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(cfg.device)

    output_dir = root_dir / f"saes/{model_name}"
    checkpoint_dir = root_dir / f"saes/{model_name}/checkpoints"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train(
        cfg=cfg,
        model=model,
        dataset=dataset,  # type: ignore
        save_path=output_dir / "sae.pt",
        log_path=output_dir,
        checkpoint_path=checkpoint_dir,
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
