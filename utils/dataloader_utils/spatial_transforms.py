import random
import math
import numbers
import collections
import numpy as np
import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageOps
import cv2

cv2.setNumThreads(0)

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if isinstance(img, Image.Image):    # if PIL Image, convert to Numpy array
            img = np.array(img)
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            if getattr(t, "randomize_parameters", None):
                t.randomize_parameters()

class ToRGB2BGR(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

class ToBGR2RGB(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class ToNormalizedTensor(object):
    def __init__(self, mean, std, norm_scale=255.0):
        self.mean = [norm_scale * m for m in mean]
        self.std = [norm_scale * s for s in std]

    def __call__(self, img):
        img = np.asarray(img, np.float32)
        img[:,:,0] = (img[:,:,0] - self.mean[0]) / self.std[0]
        img[:,:,1] = (img[:,:,1] - self.mean[1]) / self.std[1]
        img[:,:,2] = (img[:,:,2] - self.mean[2]) / self.std[2]
        return img


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.params = None

    @staticmethod
    def get_params(size, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = size[0] * size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(size[0], size[1])
        i = (size[1] - w) // 2
        j = (size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if self.params == None:
                ih, iw = img.shape[:2]
                self.params = self.get_params((iw,ih), self.scale, self.ratio)
            i,j,h,w = self.params
            img = img[i:i+h,j:j+w]
            return cv2.resize(img, self.size, self.interpolation)
        else:
            if self.params == None:
                self.params = self.get_params(img.size, self.scale, self.ratio)
            return F.resized_crop(img, *self.params, self.size, self.interpolation)

    def randomize_parameters(self):
        self.params = None


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.params = None

    @staticmethod
    def get_params(input_size, output_size):
        w, h = input_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if self.params == None:
                ih, iw = img.shape[:2]
                self.params = self.get_params((iw,ih), self.size)
            i,j,h,w = self.params
            return img[i:i+h,j:j+w]
        else:
            if self.params == None:
                self.params = self.get_params(img.size, self.size)
            return F.crop(img, *self.params)

    def randomize_parameters(self):
        self.params = None


class RandomResizedCrop2(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.params = None

    @staticmethod
    def get_params(size, scale, ratio):
        minlen = min(size[0], size[1])
        target_scale = random.uniform(*scale)
        aspect_ratio = random.uniform(*ratio)
        w = int(round(minlen * target_scale * aspect_ratio))
        h = int(round(minlen * target_scale / aspect_ratio))

        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        pad_type = -1
        if size[0] >= w:
            j = random.randint(0, size[0] - w)
        else:
            j = 0
            pad_l = random.randint(0, w - size[0])
            pad_r = w - size[0] - pad_l
            pad_type = random.randint(0,2)
        if size[1] >= h:
            i = random.randint(0, size[1] - h)
        else:
            i = 0
            pad_t = random.randint(0, h - size[1])
            pad_b = h - size[1] - pad_t
            pad_type = random.randint(0,2)
        return i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if self.params == None:
                ih, iw = img.shape[:2]
                self.params = self.get_params((iw,ih), self.scale, self.ratio)
            i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type = self.params
            if pad_type == 0:
                img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_CONSTANT,value=0)
            elif pad_type == 1:
                img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_REPLICATE)
            elif pad_type == 2:
                img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_REFLECT_101)
            img = cv2.resize(img[i:i+h,j:j+w], self.size, self.interpolation)
            return img
        else:
            if self.params == None:
                self.params = self.get_params(img.size, self.scale, self.ratio)
            i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type = self.params
            if pad_type == 0:
                img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='constant')
            elif pad_type == 1:
                img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='edge')
            elif pad_type == 2:
                img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='reflect')
            return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self):
        self.params = None

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if self.p < 0.5:
            if isinstance(img, np.ndarray):
                return cv2.flip(img, 1)
            else:
                return F.hflip(img)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class Resize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            # self.size = (size, size)
            self.size = size
            print("[CHANGED Jul 18, 2018] Now RESIZE(INT) keeps the aspect ratio of input!!!")
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        size = self.size
        if isinstance(img, np.ndarray):
            if isinstance(size, int):
                h, w = img.shape[:2]
                # if (w <= h and w == size) or (h <= w and h == size):
                #     return img
                if w < h:
                    ow = size
                    oh = int(float(size) * h / w / 2) * 2
                    return cv2.resize(img, (ow, oh), self.interpolation)
                else:
                    oh = size
                    ow = int(float(size) * w / h / 2) * 2
                    return cv2.resize(img, (ow, oh), self.interpolation)
            return cv2.resize(img, size, self.interpolation)
        else:
            return F.resize(img, size, self.interpolation)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):

        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            th, tw = self.size
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            return img[i:i+th,j:j+tw]
        else:
            return F.center_crop(img, self.size)


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        # apply transforms in HSV colorspace
        imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if self.brightness > 0 or self.contrast > 0:
            imghsv[:,:,2] = cv2.convertScaleAbs(imghsv[:,:,2], alpha=self.contrast_factor, beta=self.brightness_factor*255)
        if self.saturation > 0:
            imghsv[:,:,1] = cv2.multiply(imghsv[:,:,1], self.saturation_factor)
        if self.hue > 0:
            imghsv[:,:,0] = np.uint8((np.int16(imghsv[:,:,0]) + self.hue_factor*180) % 180)
        img_rgb = cv2.cvtColor(imghsv, cv2.COLOR_HSV2RGB)
        # blend the input image with the transformed image
        return cv2.addWeighted(img, 0.5, img_rgb, 0.5, 0)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def randomize_parameters(self):
        self.brightness_factor = np.random.uniform(-self.brightness, self.brightness)
        self.contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        self.saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        self.hue_factor = np.random.uniform(-self.hue, self.hue)


