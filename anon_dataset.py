from torch.utils.data import Dataset
from PIL import Image
import os

class AnonDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None, transform_label=None):
        """
        Args:
            annotations (pandas dataframe): dataframe with annotation scores and paths to imgs.
            root_dir (string): Directory with all the images (base folder).
        """
        self.annotations = annotations
        self.root_dir = root_dir
        self.transform = transform
        self.transform_label = transform_label


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # original image
        img1_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image1 = Image.open(img1_path).convert('RGB')
        # anonymized image
        img2_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])
        image2 = Image.open(img2_path).convert('RGB')
        # label (mean score 1-10)
        score = self.annotations.iloc[idx]['score_mean']
        
        # apply label transforms (different granularity)
        if self.transform_label:
            score = self.transform_label(score)

        # Image transforms
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Return original, anonymized, and target label
        return image1, image2, score