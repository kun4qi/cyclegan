import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageOps
from torchvision.transforms import ColorJitter


class Normalize(object):
    """Normalizes image with range of 0-255 to 0-1.
    """

    def __init__(self, min_val=0, max_val=255):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample: dict):
        normal_image = sample['normal_image']
        normal_image -= self.min_val
        normal_image /= (self.max_val - self.min_val)
        normal_image = torch.clamp(normal_image, 0, 1)

        abnormal_image = sample['normal_image']
        abnormal_image -= self.min_val
        abnormal_image /= (self.max_val - self.min_val)
        abnormal_image = torch.clamp(abnormal_image, 0, 1)

        sample.update({
            'normal_image': normal_image,
            'abnormal_image': abnormal_image
        })

        return sample


class ZScoreNormalize(object):

    def __call__(self, sample):
        normal_image = sample['normal_image']
        mean = normal_image.mean()
        std = normal_image.std()
        normal_image = normal_image.float()
        normal_image = (normal_image - mean) / std

        abnormal_image = sample['abnormal_image']
        mean = abnormal_image.mean()
        std = abnormal_image.std()
        abnormal_image = abnormal_image.float()
        abnormal_image = (abnormal_image - mean) / std

        sample.update({
            'normal_image': normal_image,
            'abnormal_image': abnormal_image
        })

        return sample


class ToImage(object):

    def __call__(self, sample):
        # assert 'label' not in sample.keys()
        normal_image = sample['normal_image']
        abnormal_image = sample['abnormal_image']

        sample.update({
            'normal_image': Image.fromarray(normal_image),
            'abnormal_image': Image.fromarray(abnormal_image)
        })

        return sample


class ToTensor(object):

    def __call__(self, sample: dict):
        normal_image = sample['normal_image']

        if type(normal_image) == Image.Image:
            normal_image = np.asarray(normal_image)

        if normal_image.ndim == 2:
            normal_image = normal_image[np.newaxis, ...]

        normal_image = torch.from_numpy(normal_image).float()

        abnormal_image = sample['abnormal_image']

        if type(abnormal_image) == Image.Image:
            abnormal_image = np.asarray(abnormal_image)

        if abnormal_image.ndim == 2:
            abnormal_image = abnormal_image[np.newaxis, ...]

        abnormal_image = torch.from_numpy(abnormal_image).float()
        sample.update({
            'normal_image': normal_image,
            'abnormal_image': abnormal_image
        })

        if 'normal_label' in sample.keys():
            normal_label = sample['normal_label']

            if normal_label.ndim == 2:
                normal_label = normal_label[np.newaxis, ...]

            normal_label = torch.from_numpy(normal_label).int()
            sample.update({
            'normal_label': normal_label
            })
        
        if 'abnormal_label' in sample.keys():
            abnormal_label = sample['abnormal_label']

            if abnormal_label.ndim == 2:
                abnormal_label = abnormal_label[np.newaxis, ...]

            abnormal_label = torch.from_numpy(abnormal_label).int()
    
            sample.update({
                'abnormal_label': abnormal_label
            })
        return sample


class RandomHorizontalFlip(object):

    def __call__(self, sample: dict):
        assert 'normal_label' not in sample.keys()
        assert 'abnormal_label' not in sample.keys()

        normal_image = sample['normal_image']
        if random.random() < 0.5:
            normal_image = normal_image.transpose(Image.FLIP_LEFT_RIGHT)

        abnormal_image = sample['abnormal_image']
        if random.random() < 0.5:
            abnormal_image = abnormal_image.transpose(Image.FLIP_LEFT_RIGHT)

        sample.update({
            'normal_image': normal_image,
            'abnormal_image': abnormal_image,
        })

        return sample


class RandomVerticalFlip(object):

    def __call__(self, sample: dict):
        assert 'normal_label' not in sample.keys()
        assert 'abnormal_label' not in sample.keys()

        normal_image = sample['normal_image']
        if random.random() < 0.5:
            normal_image = normal_image.transpose(Image.FLIP_TOP_BOTTOM)

        abnormal_image = sample['abnormal_image']
        if random.random() < 0.5:
            abnormal_image = abnormal_image.transpose(Image.FLIP_TOP_BOTTOM)

        sample.update({
            'normal_image': normal_image,
            'abnormal_image': abnormal_image
        })

        return sample


class RandomRotate(object):

    def __init__(self, degree=20):
        self.degree = degree

    def __call__(self, sample: dict):
        assert 'normal_label' not in sample.keys()
        assert 'abnormal_label' not in sample.keys()

        normal_image = sample['normal_image']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        normal_image = normal_image.rotate(rotate_degree, Image.BILINEAR)

        abnormal_image = sample['abnormal_image']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        abnormal_image = abnormal_image.rotate(rotate_degree, Image.BILINEAR)

        sample.update({
            'normal_image': normal_image,
            'abnormal_image': abnormal_image,
        })

        return sample


class RandomScale(object):

    def __init__(self, mean=1.0, var=0.05, image_fill=0):
        self.mean = mean
        self.var = var
        self.image_fill = image_fill

    def __call__(self, sample: dict):
        assert 'normal_label' not in sample.keys()
        assert 'abnormal_label' not in sample.keys()

        normal_image = sample['normal_image']
        base_size = normal_image.size
        scale_factor = random.normalvariate(self.mean, self.var)
        size = (
            int(base_size[0] * scale_factor),
            int(base_size[1] * scale_factor),
        )
        normal_image = normal_image.resize(size, Image.BILINEAR)
        if scale_factor < 1.0:
            pad_h = base_size[0] - normal_image.size[0]
            pad_w = base_size[1] - normal_image.size[1]
            ori_h = random.randint(0, pad_h)
            ori_w = random.randint(0, pad_w)
            normal_image = ImageOps.expand(
                normal_image,
                border=(ori_h, ori_w, pad_h - ori_h, pad_w - ori_w),
                fill=self.image_fill
            )
        else:
            ori_h = random.randint(0, normal_image.size[0] - base_size[0])
            ori_w = random.randint(0, normal_image.size[1] - base_size[1])
            normal_image = normal_image.crop((
                ori_h, ori_w,
                ori_h + base_size[0], ori_w + base_size[1]
            ))

        abnormal_image = sample['abnormal_image']
        base_size = abnormal_image.size
        scale_factor = random.normalvariate(self.mean, self.var)
        size = (
            int(base_size[0] * scale_factor),
            int(base_size[1] * scale_factor),
        )
        abnormal_image = abnormal_image.resize(size, Image.BILINEAR)
        if scale_factor < 1.0:
            pad_h = base_size[0] - abnormal_image.size[0]
            pad_w = base_size[1] - abnormal_image.size[1]
            ori_h = random.randint(0, pad_h)
            ori_w = random.randint(0, pad_w)
            abnormal_image = ImageOps.expand(
                abnormal_image,
                border=(ori_h, ori_w, pad_h - ori_h, pad_w - ori_w),
                fill=self.image_fill
            )
        else:
            ori_h = random.randint(0, abnormal_image.size[0] - base_size[0])
            ori_w = random.randint(0, abnormal_image.size[1] - base_size[1])
            abnormal_image = abnormal_image.crop((
                ori_h, ori_w,
                ori_h + base_size[0], ori_w + base_size[1]
            ))

        sample.update({
            'normal_image': normal_image,
            'abnormal_image': abnormal_image
        })
        return sample


class RandomColorJitter(object):
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3):
        self.filter = ColorJitter(brightness, contrast, saturation)

    def __call__(self, sample: dict):
        normal_image = sample['normal_image']
        normal_image = normal_image.convert('RGB')
        normal_image = self.filter(normal_image)
        normal_image = normal_image.convert('L')

        abnormal_image = sample['abnormal_image']
        abnormal_image = abnormal_image.convert('RGB')
        abnormal_image = self.filter(abnormal_image)
        abnormal_image = abnormal_image.convert('L')

        sample.update({
            'normal_image': normal_image,
            'abnormal_image': abnormal_image
        })

        return sample


class RandomSliceSelect(object):
    def __init__(self, threshold=1, max_iter=10):
        self.threshold = threshold
        self.max_iter = max_iter

    def __call__(self, sample: dict):
        normal_image = sample['normal_image']
        z_max = normal_image.shape[2]
        mean = 0.0
        n_iter = 0
        while n_iter < self.max_iter:
            selected_z = random.randint(0, z_max - 1)
            normal_selected = normal_image[..., selected_z]
            mean = np.mean(normal_selected)
            if mean > self.threshold:
                break
            n_iter += 1

        abnormal_image = sample['abnormal_image']
        z_max = abnormal_image.shape[2]
        mean = 0.0
        n_iter = 0
        while n_iter < self.max_iter:
            selected_z = random.randint(0, z_max - 1)
            abnormal_selected = abnormal_image[..., selected_z]
            mean = np.mean(abnormal_selected)
            if mean > self.threshold:
                break
            n_iter += 1

        sample.update({
            'normal_image': normal_selected,
            'abnormal_image': abnormal_selected
        })

        return sample
