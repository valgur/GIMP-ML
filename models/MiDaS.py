import sys

import numpy as np
import torch
import torch.hub
from PIL import Image
from torchvision.transforms import Compose

from _model_base import ModelBase, handle_alpha
from _util import apply_colormap, to_rgb


# Simplified transforms from
# https://github.com/intel-isl/MiDaS/blob/master/models/transforms.py
class Resize:
    def __init__(self, width, height, image_interpolation_method=Image.BICUBIC):
        self.__width = width
        self.__height = height
        self.__multiple_of = 32
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x):
        return (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

    def get_size(self, width, height):
        scale_height = self.__height / height
        scale_width = self.__width / width

        # scale such that output size is upper bound
        if scale_width < scale_height:
            # fit width
            scale_height = scale_width
        else:
            # fit height
            scale_width = scale_height

        new_height = self.constrain_to_multiple_of(scale_height * height)
        new_width = self.constrain_to_multiple_of(scale_width * width)
        return new_width, new_height

    def __call__(self, image):
        width, height = self.get_size(image.shape[1], image.shape[0])
        resized = Image.fromarray(image).resize((width, height), self.__image_interpolation_method)
        return np.array(resized)


class NormalizeImage:
    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, image):
        return (image - self.__mean) / self.__std


class PrepareForNet:
    def __call__(self, image):
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image, dtype=np.float32)
        tensor = torch.from_numpy(image)
        return tensor.unsqueeze(0)


class MiDaS(ModelBase):
    def __init__(self):
        super().__init__()
        self.hub_repo = "intel-isl/MiDaS"

    def load_model(self):
        model = torch.hub.load(self.hub_repo, "MiDaS", pretrained=True)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def get_transform():
        return Compose([
            Resize(384, 384),
            lambda x: x / 255.,
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet()
        ])

    @handle_alpha
    @torch.no_grad()
    def predict(self, input_image, colormap=None):
        h, w, d = input_image.shape
        assert d == 3, "Input image must be RGB"

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        transform = self.get_transform()
        image_tensor = transform(input_image).to(self.device)
        prediction = self.model.forward(image_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        disp = prediction.squeeze().cpu().numpy()
        disp /= disp.max()

        if colormap:
            out = apply_colormap(disp, colormap)
        else:
            out = to_rgb(disp)
        return (out * 255).astype(np.uint8)


model = MiDaS()

if __name__ == '__main__':
    rpc_url = sys.argv[1]
    model.process_rpc(rpc_url)
