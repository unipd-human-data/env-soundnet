#SNN
class SNNClassifier(torch.nn.Module):
    def __init__(
        self,
        n_mels,
        hidden_sizes,
        num_classes,
        surr_grad,
        learn_thr=True,
        learn_beta=True,
    ):
        super(SNNClassifier, self).__init__()
        self.n_mels = n_mels
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.surr_grad = surr_grad
        self.learn_thr = learn_thr
        self.learn_beta = learn_beta

        # Layer 1: Input to Hidden 1
        self.fc1 = torch.nn.Linear(n_mels, self.hidden_sizes[0])
        self.lif1 = snn.Leaky(
            beta=torch.full((self.hidden_sizes[0],), 0.5),
            learn_beta=learn_beta,
            learn_threshold=learn_thr,
            spike_grad=surr_grad,
            reset_mechanism="zero",
        )

        # Layer 2: Hidden 1 to Hidden 2
        self.fc2 = torch.nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.lif2 = snn.Leaky(
            beta=torch.full((self.hidden_sizes[1],), 0.5),
            learn_beta=learn_beta,
            learn_threshold=learn_thr,
            spike_grad=surr_grad,
            reset_mechanism="zero",
        )

        # Layer 3: Hidden 2 to Hidden 3
        self.fc3 = torch.nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.lif3 = snn.Leaky(
            beta=torch.full((self.hidden_sizes[2],), 0.5),
            learn_beta=learn_beta,
            learn_threshold=learn_thr,
            spike_grad=surr_grad,
            reset_mechanism="zero",
        )

        # Output Layer
        self.fc_out = torch.nn.Linear(self.hidden_sizes[2], num_classes)
        self.lif_out = snn.Leaky(
            beta=torch.full((num_classes,), 0.5),
            learn_beta=learn_beta,
            learn_threshold=learn_thr,
            spike_grad=surr_grad,
            reset_mechanism='zero',
        )

    def forward(self, x):
        batch_size, time_steps, _ = x.shape

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        mem_rec = []

        for step in range(time_steps):
            x_t = x[:, step, :]

            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur_out = self.fc_out(spk3)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)


#SCNN
class C_SNN(torch.nn.Module):
    def __init__(
        self,
        surr_grad,
        n_mels: int = 128,
        num_classes: int = 10,
        conv_channels1: int = 8,
        kernel_size: int = 3,
        pool_kernel: int = 2,
        dropout_rate: float = 0.15
    ):
        super().__init__()
        self.surr_grad = surr_grad

        # --- First Spiking Convolutional Block ---
        self.conv1 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.lif1 = snn.Leaky(
            beta=0.9,
            threshold=1.0,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=surr_grad
        )
        self.pool1 = torch.nn.MaxPool1d(kernel_size=pool_kernel)

        # --- Fully Connected Layer ---
        freq_after_pool = n_mels // pool_kernel
        self.flattened_size = conv_channels1 * freq_after_pool

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc1 = torch.nn.Linear(self.flattened_size, num_classes, bias=False)
        self.lif_out = snn.Leaky(
            beta=0.9,
            threshold=1.0,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=surr_grad
        )

    def forward(self, spikes: torch.Tensor):
        """
        spikes: [batch_size, time_steps, n_mels]
        Returns:
            spk_rec: [time_steps, batch_size, num_classes]
            mem_rec: [time_steps, batch_size, num_classes]
        """
        B, T, F = spikes.shape

        mem1 = self.lif1.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        mem_rec = []

        spikes_tm = spikes.permute(1, 0, 2)  # [T, B, F]

        for t in range(T):
            x_t = spikes_tm[t].unsqueeze(1)  # [B, 1, F]

            # --- Conv Block 1 ---
            x = self.conv1(x_t)
            spk1, mem1 = self.lif1(x, mem1)
            x = self.pool1(spk1)

            # --- Flatten + FC ---
            x = x.view(B, -1)
            x = self.dropout(x)
            x = self.fc1(x)
            spk_out, mem_out = self.lif_out(x, mem_out)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        spk_rec = torch.stack(spk_rec, dim=0)
        mem_rec = torch.stack(mem_rec, dim=0)

        return spk_rec, mem_rec
