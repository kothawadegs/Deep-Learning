import torch
import torchvision
from torchvision import datasets
import albumentations
from albumentations.pytorch import ToTensor
import numpy as np

'''
basic transformation to make albumentations data processing compatible with pytorch DataLoader
'''
class AlbumCompose():
    def __init__(self, transform=None):
        self.transform = transform
        
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img
		
'''
CIFAR10 dataset
'''
def get_dataset(train_transforms, test_transforms):
	trainset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
	testset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
	return trainset, testset
	
'''
calculate normalized mean and std for entire dataset
'''
def calculate_dataset_mean_std():
	data_transforms = albumentations.Compose([
										  ToTensor()
										  ])
										  
	trainset, testset = get_dataset(data_transforms,data_transforms)
	
	data = np.concatenate([trainset.data, testset.data], axis=0)
	data = data.astype(np.float32)/255.

	print("\nTotal dataset(train+test) shape: ", data.shape)

	means = []
	stdevs = []
	for i in range(3): # 3 channels
		pixels = data[:,:,:,i].ravel()
		means.append(np.mean(pixels))
		stdevs.append(np.std(pixels))

	return [means[0], means[1], means[2]], [stdevs[0], stdevs[1], stdevs[2]]

'''
args: contains dictionary for type of image aumentation to be applied
Example:
{
	'GeneralParams': {'input_size': 32},
	'CoarseDropout': {'apply': True,
					  'fill_value': [125.42650446295738, 123.07660415768623, 114.03038397431374],
					  'max_height': 16,
					  'max_holes': 1,
					  'max_width': 16,
					  'min_height': 4,
					  'min_width': 4,
					  'p': 0.2},
	'ElasticTransform': {'apply': True,
						'alpha': 1,
						'alpha_affine': 10,
						'sigma': 50,
						'p': 0.2
						},
 
	'HorizontalFlip': {'apply': True, 'p': 1.0},
	'RandomCrop': {'apply': True, 'p': 0.2, 'size': 8},
	'Normalize': {'apply': True,
				  'means': [0.49186864, 0.48265335, 0.44717798],
				  'stds': [0.24697131, 0.24338901, 0.26159254]}
}
'''
def get_data_transform(args):
    #print(args)

    # read general parameters
    key = "GeneralParams"
    if key not in args:
      print("Mandatory {} attribute is missing".format(key))
      return None, None
    
    inp_size = args[key]["input_size"]

    # prepare Augmentation: Normalized
    key = "Normalize"
    if key not in args:
      print("Mandatory {} attribute is missing".format(key))
      return None, None

    normalize = albumentations.Normalize(mean=args[key]["means"], std=args[key]["stds"])
    resize = albumentations.Resize(inp_size,inp_size)

    # prepare Augmentation: CoarseDropout (same as cutout)
    cutout = None
    key = "CoarseDropout"
    if key in args and args[key]["apply"]:
      print("{}/Cutout is enabled".format(key))
      cutout = albumentations.CoarseDropout(
                          max_holes=args[key]["max_holes"], 
                          max_height=args[key]["max_height"], max_width=args[key]["max_width"], 
                          min_height=args[key]["min_height"], min_width=args[key]["min_width"], 
                          fill_value=args[key]["fill_value"],
                          p=args[key]["p"])
    
    # prepare Augmentation: ElasticTransform 
    elasticTransform = None
    key = "ElasticTransform"
    if key in args and args[key]["apply"]:
      print("{} is enabled".format(key))
      elasticTransform = albumentations.ElasticTransform(
                          sigma=args[key]["sigma"], 
                          alpha=args[key]["alpha"], 
                          alpha_affine=args[key]["alpha_affine"],
                          p=args[key]["p"])
    
    # prepare Augmentation: HorizontalFlip 
    horizontalFlip = None
    key = "HorizontalFlip"
    if key in args and args[key]["apply"]:
      print("{} is enabled".format(key))
      horizontalFlip = albumentations.HorizontalFlip(p=args[key]["p"])

    # prepare Augmentation: RandomCrop 
    randomCrop = None
    key = "RandomCrop"
    if key in args and args[key]["apply"]:
      print("{} is enabled".format(key))
      size = args[key]["size"]
      randomCrop = albumentations.RandomCrop(size, size,
                                              p=args[key]["p"])
    
    # prepare train transform list
    train_transform_list = []

    # arrange all the transform in required order
    if randomCrop is not None:
      train_transform_list.append(randomCrop)
      train_transform_list.append(resize)
    
    if horizontalFlip is not None:
      train_transform_list.append(horizontalFlip)

    if elasticTransform is not None:
      train_transform_list.append(elasticTransform)

    if cutout is not None:
      train_transform_list.append(cutout)

    train_transform_list.append(normalize)
    train_transform_list.append(ToTensor())

    train_transforms = albumentations.Compose(train_transform_list)
    train_transforms = AlbumCompose(train_transforms)

    # Test Phase transformations
    test_transforms = albumentations.Compose([
                                          #resize,
                                          normalize,
                                          ToTensor()
                                          ])
    
    test_transforms = AlbumCompose(test_transforms)
    
    return train_transforms, test_transforms

'''
preparing data loader using pytorch DataLoader
'''
def get_dataloader(train_transforms, test_transforms, batch_size=32, num_workers=1):

	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)

	# dataloader arguments - something you'll fetch these from cmdprmt
	dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=8)

	trainset, testset = get_dataset(train_transforms, test_transforms)

	# train dataloader
	train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

	# test dataloader
	test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

	return train_loader, test_loader