from albumentations import (
	Compose,
    HorizontalFlip,
    Normalize,
    CoarseDropout,
    RandomCrop,
    CenterCrop,
    PadIfNeeded,
    OneOf,
)
from albumentations.pytorch import ToTensor
import numpy as np
from cv2 import BORDER_CONSTANT, BORDER_REFLECT

def albumentations_transforms(p=1.0, is_train=False):
	# Mean and standard deviation of train dataset
	mean = np.array([0.4914, 0.4822, 0.4465])
	std = np.array([0.2023, 0.1994, 0.2010])
	transforms_list = []
	# Use data aug only for train data
	if is_train:
		transforms_list.extend([
			PadIfNeeded(min_height=40, min_width=40, border_mode=BORDER_CONSTANT,
					value=mean*255.0, p=1.0),
			OneOf([
				RandomCrop(height=32, width=32, p=0.8),
				CenterCrop(height=32, width=32, p=0.2),
			], p=1.0),
			HorizontalFlip(p=0.5),
			CoarseDropout(max_holes=1, max_height=8, max_width=8, min_height=8,
						min_width=8, fill_value=mean*255.0, p=0.75),

		])
	transforms_list.extend([
		Normalize(
			mean=mean,
			std=std,
			max_pixel_value=255.0,
			p=1.0
		),
		ToTensor()
	])
	transforms = Compose(transforms_list, p=p)
	return lambda img:transforms(image=np.array(img))["image"]
