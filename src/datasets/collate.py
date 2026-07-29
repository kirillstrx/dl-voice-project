import torch


def collate_fn(dataset_items: list[dict]):
    if "audio" in dataset_items[0]:
        return {
            "audio": torch.stack([item["audio"] for item in dataset_items]),
            "labels": torch.tensor(
                [item["labels"] for item in dataset_items],
                dtype=torch.long,
            ),
            "audio_id": [item["audio_id"] for item in dataset_items],
        }

    return {
        "data_object": torch.vstack([item["data_object"] for item in dataset_items]),
        "labels": torch.tensor([item["labels"] for item in dataset_items]),
    }
