import torch
from torch import nn


class LogPowerSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=1724,
        win_length=1724,
        hop_length=130,
        log_epsilon=1e-6,
    ):
        """
        Args:
            n_fft (int): FFT size.
            win_length (int): window length.
            hop_length (int): hop length.
            log_epsilon (float): minimum value before
                logarithm.
        """
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.log_epsilon = log_epsilon

        self.register_buffer(
            "window",
            torch.blackman_window(win_length),
            persistent=False,
        )

    def forward(self, audio):
        """
        Convert waveform into spectrogram.

        Args:
            audio (Tensor): input waveform.

        Returns:
            spectrogram (Tensor): log-power spectrogram.
        """
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )

        power = spectrum.abs().pow(2)

        spectrogram = torch.log(power.clamp_min(self.log_epsilon))

        return spectrogram
