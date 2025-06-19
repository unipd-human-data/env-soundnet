import torch
from torch.utils.data import Dataset
from snntorch import spikegen
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#DELTA ENCODING DATASET
class DeltaAudioDataset(Dataset):
    def __init__(self, X, y, threshold=0.05, timesteps=10, off_spike=True):
        """
        X: Tensor of shape [num_samples, n_mels, time_steps, 1]
        y: Tensor of shape [num_samples]
        """
        self.X = X
        self.y = y
        self.timesteps = timesteps
        self.threshold = threshold
        self.off_spike = off_spike

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].squeeze(-1)  # [n_mels, time_steps]
        y = self.y[idx]

        chunks = torch.tensor_split(x, self.timesteps, dim=-1)

        x_chunks = torch.stack(
            [window.mean(dim=-1) for window in chunks], dim=0
        )

        # Apply delta modulation (shape stays [time_steps, n_mels])
        spike_train = spikegen.delta(
            x_chunks, threshold=self.threshold, off_spike=self.off_spike
        )

        return spike_train, y.long()
    
#TAE ENCODING DATASET
class ThresholdAdaptiveDataset(Dataset):

  def __init__(self, X, y, alpha=0.9, threshold=0.05):
    self.X = X
    self.y = y
    self.alpha = alpha
    self.threshold = threshold

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    x = self.X[idx].squeeze(-1)  # [n_mels, time_steps]
    y = self.y[idx]

    spike_train = torch.zeros(
        x.shape[1], x.shape[0], dtype=torch.int8
    )  # [time_steps, n_mels]

    for mel_idx in range(x.shape[0]):
      spike_train[:, mel_idx] = self._encode_tae(x[mel_idx])

    return spike_train.to(torch.float32), y.long()

  def _encode_tae(self, signal):
    if signal.dim() == 2:
        signal = signal.squeeze(0) 

    signal_np = signal.numpy()
    spikes = np.zeros_like(signal_np, dtype=np.int8)

    base = signal_np[0]
    threshold = self.threshold  
    alpha = self.alpha          

    for i in range(1, len(signal_np)):
        diff = signal_np[i] - base
        if diff >= threshold:
            spikes[i] = 1
            base += threshold
            threshold *= alpha
        elif diff <= -threshold:
            spikes[i] = -1
            base -= threshold
            threshold *= alpha
        else:
            threshold *= alpha

    return torch.from_numpy(spikes)
  
#STEP FORWARD ENCODING DATASET
class StepForwardDataset(Dataset):
  def __init__(self, X, y, threshold=0.05):
    self.X = X
    self.y = y
    self.threshold = threshold

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    x = self.X[idx].squeeze(-1)  # [n_mels, time_steps]
    y = self.y[idx]

    spike_train = torch.zeros(
        x.shape[1], x.shape[0], dtype=torch.int8
    )  # [time_steps, n_mels]

    for mel_idx in range(x.shape[0]):
        spikes, _ = self._encode(x[mel_idx])
        spike_train[:, mel_idx] = torch.from_numpy(spikes)

    return spike_train, y.long()

  def _encode(self, input_signal):
    if isinstance(input_signal, torch.Tensor):
        input_signal = input_signal.numpy()

    L = len(input_signal)
    spikes = np.zeros(L, dtype=np.int8)
    base_history = np.zeros(L) 

    base = input_signal[0]
    base_history[0] = base

    for i in range(1, L):
        if input_signal[i] > base + self.threshold:
            spikes[i] = 1
            base = base + self.threshold  
        elif input_signal[i] < base - self.threshold:
            spikes[i] = -1
            base = base - self.threshold  
        base_history[i] = base

    return spikes, base_history
  
# Visualization Utility
def visualize_spike_trains(spike_tensor, sample_idx=0, mel_bin=10):

    if isinstance(spike_tensor, list):
        spikes = spike_tensor[0].squeeze(-1)[sample_idx].detach().cpu().numpy()
    elif isinstance(spike_tensor, torch.Tensor):
        # If 4D tensor [batch_size, time_frames, n_mels]
        if len(spike_tensor.shape) == 3:
            spikes = spike_tensor.squeeze(-1)[sample_idx, :, :].detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected spike tensor shape: {spike_tensor.shape}")
    else:
        raise TypeError("spike_tensor must be a torch.Tensor or a list containing tensors")

    if len(spikes.shape) > 2:
        spikes = spikes.squeeze()

    pos_spikes = (spikes > 0).astype(float)
    neg_spikes = (spikes < 0).astype(float)

    plt.figure(figsize=(10, 12))

    # 1. Raster Plot with Inverted Y-axis
    plt.subplot(3, 1, 1)
    plt.imshow(spikes.T, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1, origin='lower')
    plt.colorbar(ticks=[-1, 0, 1], label='Spike Type')
    plt.title("Full Raster Plot")
    plt.xlabel("Time Step")
    plt.ylabel("Mel Bin")
    num_mel_bins = spikes.shape[1]
    tick_interval = max(1, num_mel_bins // 7)
    plt.yticks(np.arange(0, num_mel_bins, tick_interval))

    # 2. Single Mel Bin Spike Train with vlines
    plt.subplot(3, 1, 2)
    pos_times = np.where(spikes[:, mel_bin] == 1)[0]
    neg_times = np.where(spikes[:, mel_bin] == -1)[0]
    plt.vlines(pos_times, 0, 1, color='red', linewidth=0.8)
    plt.vlines(neg_times, -1, 0, color='blue', linewidth=0.8)
    plt.yticks([-1, 0, 1])
    plt.ylim(-1.2, 1.2)
    plt.title(f"Spike Train for Mel Bin {mel_bin}")
    plt.xlabel("Time Step")
    plt.ylabel("Spike Value")
    plt.grid(False)
    legend_elements = [plt.Line2D([0], [0], color='red', lw=2, label='Positive Spikes'),
                      plt.Line2D([0], [0], color='blue', lw=2, label='Negative Spikes')]
    plt.legend(handles=legend_elements)

    # 3. Density Plot with Legend
    plt.subplot(3, 1, 3)
    plt.stackplot(np.arange(spikes.shape[0]),
                  pos_spikes.sum(axis=1),
                  -neg_spikes.sum(axis=1),
                  colors=['red', 'blue'])

    plt.legend(['Positive Spikes', 'Negative Spikes'])
    plt.title("Spike Polarity Balance Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Net Spike Count")

    plt.tight_layout()
    plt.show()