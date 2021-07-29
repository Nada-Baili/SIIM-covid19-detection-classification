import cv2, torch, os
from albumentations import Compose, Normalize, Resize, HorizontalFlip, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2

def get_transforms(*, data):
    assert data in ('train', 'valid')

    if data == 'train':
        return Compose([
            Resize(256,256),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(256,256),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

class Data:
    def __init__(self, params, loss, images, labels=None, transform=None, is_Test=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.is_Test = is_Test
        self.params = params
        self.loss = loss

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        file_name = self.images[idx]
        if not self.is_Test:
            file_path = os.path.join(self.params["train_images_path"], file_name+'.jpg')
        else:
            file_path = os.path.join(self.params["test_images_path"], file_name+'.jpg')
        image = cv2.imread(file_path)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if not self.is_Test:
            label = self.labels[idx]
            if self.loss == "FOCAL":
                target = torch.zeros(4)
                target[int(label)] = 1
                label = target
            
            return image, label

        return image