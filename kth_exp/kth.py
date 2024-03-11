import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import glob

class KTHDataset(Dataset):
    def __init__(self, data_root='./KTH/data/kth/processed/', seq_len=3, train=True):
        self.data_root = data_root
        self.seq_len = seq_len
        self.transform = transforms.Compose([
                                             transforms.ToTensor()]) #transforms.Grayscale(num_output_channels=1),
        self.classes = ['jogging', 'running', 'walking']
        self.samples = []
        self.train = train
        
        if self.train:
            persons = list(range(1, 21))
        else:
            persons = list(range(21, 26))

        for c in self.classes:
            c_dir = os.path.join(data_root, c)
            for p in persons:
                p_dir = os.path.join(c_dir, 'person{:02d}'.format(p))
                for vid_dir in glob.glob(p_dir+"*d4"):
                    #vid_dir = os.path.join(p_dir, vid_name)
                    num_frames = len(os.listdir(vid_dir))
                    for i in range(num_frames - seq_len + 1):
                        sample = {
                            'dir': vid_dir,
                            'frame_indices': list(range(i, i + seq_len))
                        }
                        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        dir = sample['dir']
        frame_indices = sample['frame_indices']

        frames = []
        frame_paths = []
        for i in frame_indices:
            frame_path = os.path.join(dir, f"image-{i+1:03d}_64x64.png")
            frame_paths.append(frame_path)
            frame = Image.open(frame_path).convert('RGB')
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)

        frames = torch.stack(frames, dim=0)

        return frames, frame_paths
