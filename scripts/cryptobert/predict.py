import argparse
from pathlib import Path

import torch
from saefarer.model import SAE
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification

from datasets import load_from_disk


def main(root_dir):
    """Train the SAE"""

    root_dir = Path(root_dir)

    device = "mps"

    model_name = "ElKulako/cryptobert"

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)

    sae = SAE.load(
        root_dir / f"saes/{model_name}/sae.pt",
        device,
    )

    dataset = load_from_disk(
        (root_dir / "datasets/ElKulako/stocktwits-crypto_tokenized").as_posix()
    )

    output_path = root_dir / f"saes/{model_name}/predictions.json"

    sae_mean_activations = []
    labels = []

    for i, x in enumerate(dataset.select(range(32 * 300)).iter(batch_size=32)):
        if i % 50 == 0:
            print(i)
        labels.append(x["label"].cpu().detach())
        output = model(
            x["input_ids"].to(device),
            output_hidden_states=True,
        )
        batch_acts = output.hidden_states[10]
        batch_sae_acts, _ = sae.encode(batch_acts)
        mean_batch_sae_acts = batch_sae_acts.mean(dim=1)
        sae_mean_activations.append(mean_batch_sae_acts.cpu().detach())

    X = torch.concat(sae_mean_activations).numpy()
    y = torch.concat(labels).numpy()

    clf = LogisticRegression(random_state=0).fit(X, y)
    y_pred = clf.predict(X)
    accuracy = (y == y_pred).sum() / y.shape[0]

    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        default="../..",
        help="path to directory containing datasets and models",
    )
    args = parser.parse_args()

    main(args.path)
