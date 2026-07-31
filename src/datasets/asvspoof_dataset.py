import math
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset


PROTOCOL_NAMES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


class ASVspoofDataset(BaseDataset):
    def __init__(
        self,
        root,
        split,
        sample_rate=16000,
        num_samples=95840,
        *args,
        **kwargs,
    ):
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = num_samples

        super().__init__(
            index=self._create_index(),
            *args,
            **kwargs,
        )

    def _create_index(self):
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
        if self.split not in PROTOCOL_NAMES:
            raise ValueError(
                f"Unknown split: {self.split}"
            )

        protocol_name = PROTOCOL_NAMES[self.split]
        protocol_paths = list(
            self.root.rglob(protocol_name)
        )

        if not protocol_paths:
            raise FileNotFoundError(
                f"{protocol_name} was not found in {self.root}"
            )

        return protocol_paths[0]

    def _find_audio_dir(self):
        directory_name = f"ASVspoof2019_LA_{self.split}"

        for dataset_dir in self.root.rglob(directory_name):
            audio_dir = dataset_dir / "flac"

            if audio_dir.is_dir():
                return audio_dir

        raise FileNotFoundError(
            f"Audio directory for {self.split} "
            f"was not found in {self.root}"
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

        if audio_length > self.num_samples:
            start = (
                audio_length - self.num_samples
            ) // 2

            audio = audio[
                start : start + self.num_samples
            ]

        elif audio_length < self.num_samples:
            repeats = math.ceil(
                self.num_samples / audio_length
            )

            audio = audio.repeat(repeats)
            audio = audio[: self.num_samples]

        return audio

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        audio = self.load_object(
            data_dict["path"]
        )
        audio = self._fix_audio_length(audio)

        return self.preprocess_data(
            {
                "audio": audio,
                "labels": data_dict["label"],
                "audio_id": data_dict["audio_id"],
            }
        )
    