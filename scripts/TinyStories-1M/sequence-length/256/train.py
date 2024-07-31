import argparse
from pathlib import Path

from saefarer.config import Config
from saefarer.training import train
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_from_disk


def main(root_dir):
    """Train the SAE"""

    root_dir = Path(root_dir)

    cfg = Config(
        device="cuda",
        dtype="float32",
        # dataset
        is_dataset_tokenized=True,
        prepend_bos_token=True,
        dataset_column="input_ids",
        # dimensions
        d_sae=256,
        d_in=64,
        # loss functions
        k=4,
        aux_k=32,
        aux_k_coef=1 / 32,
        dead_tokens_threshold=10_000_000,
        hidden_state_index=7,
        normalize=False,
        # batch sizes
        model_sequence_length=256,
        model_batch_size_sequences=32,
        n_batches_in_store=64,
        sae_batch_size_tokens=4096,
        # adam
        lr=3e-4,
        beta1=0.9,
        beta2=0.999,
        eps=6.25e-10,
        # training
        total_training_tokens=200_000_000,
        # logging
        logger="wandb",
        log_batch_freq=500,
        wandb_project="saefarer",
        wandb_group="TinyStories-1M_sequence-length",
        wandb_name="256",
        wandb_notes="Comparing execution time for 128 vs. 256 seq len.",
    )

    model = AutoModelForCausalLM.from_pretrained(
        root_dir / "models/roneneldan/TinyStories-1M"
    )
    model.to(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(
        root_dir / "models/EleutherAI/gpt-neo-125M"
    )

    dataset = load_from_disk(
        (root_dir / "datasets/roneneldan/TinyStories_tokenized_256").as_posix()
    )

    output_dir = root_dir / "saes/roneneldan/TinyStories-1M/sequence-length/256"
    output_dir.mkdir(parents=True, exist_ok=True)

    train(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        save_path=output_dir / "sae.pt",
        log_path=output_dir,
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
