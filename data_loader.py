class TokenDataset(TorchDataset):
    def __init__(self, pt_file_path):
        self.samples = torch.load(pt_file_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]