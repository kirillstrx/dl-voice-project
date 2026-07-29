import random
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from src.datasets.base_dataset import BaseDataset

PROTOCOL_NAMES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


class ASVspoofDataset(BaseDataset):
    """
    ASVspoof 2019 Logical Access dataset.
    """

    def __init__(
        self,
        root,
        split,
        sample_rate=16000,
        num_samples=79594,
        *args,
        **kwargs,
    ):
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = num_samples

        index = self._create_index()

        super().__init__(
            index=index,
            *args,
            **kwargs,
        )

    def _create_index(self):
        """
        Create dataset index from the protocol file.

        Returns:
            index (list[dict]): dataset index.
        """
        protocol_path = self._find_protocol_path()
        audio_dir = self._find_audio_dir()

        index = []

        with protocol_path.open("r") as protocol:
            for line in protocol:
                _, audio_id, _, _, label = line.strip().split()

                index.append(
                    {
                        "path": str(audio_dir / f"{audio_id}.flac"),
                        "label": 1 if label == "bonafide" else 0,
                        "audio_id": audio_id,
                    }
                )

        return index

    def _find_protocol_path(self):
        """
        Find the protocol file inside the dataset directory.

        Returns:
            protocol_path (Path): path to the protocol file.
        """
        if self.split not in PROTOCOL_NAMES:
            raise ValueError(
                f"Unknown split: {self.split}. " "Expected train, dev or eval."
            )

        protocol_name = PROTOCOL_NAMES[self.split]
        protocol_paths = list(self.root.rglob(protocol_name))

        if len(protocol_paths) == 0:
            raise FileNotFoundError(
                f"Protocol file {protocol_name} " f"was not found in {self.root}."
            )

        return protocol_paths[0]

    def _find_audio_dir(self):
        directory_name = f"ASVspoof2019_LA_{self.split}"

        for dataset_dir in self.root.rglob(directory_name):
            audio_dir = dataset_dir / "flac"

            if audio_dir.is_dir():
                return audio_dir

        raise FileNotFoundError(
            f"Audio directory for split "
            f"{self.split} was not found "
            f"in {self.root}."
        )

    def load_object(self, path):
        audio, sample_rate = sf.read(
            path,
            dtype="float32",
        )

        audio = torch.from_numpy(audio)

        if audio.ndim == 2:
            audio = audio.mean(dim=1)

        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )

        return audio

    def _fix_audio_length(self, audio):
        audio_length = audio.shape[-1]

        if audio_length < self.num_samples:
            padding = self.num_samples - audio_length
            audio = F.pad(
                audio,
                (0, padding),
            )

        elif audio_length > self.num_samples:
            max_start = audio_length - self.num_samples

            if self.split == "train":
                start = random.randint(
                    0,
                    max_start,
                )
            else:
                start = max_start // 2

            audio = audio[start : start + self.num_samples]

        return audio

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        audio = self.load_object(data_dict["path"])
        audio = self._fix_audio_length(audio)

        instance_data = {
            "audio": audio,
            "labels": data_dict["label"],
            "audio_id": data_dict["audio_id"],
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data
