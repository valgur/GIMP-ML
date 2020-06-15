import sys

import numpy as np
import torch
import torch.hub
from PIL import Image
from torchvision import transforms

from _model_base import ModelBase, handle_alpha
from _util import to_rgb, apply_colormap


class Monodepth2(ModelBase):
    def __init__(self):
        super().__init__()
        self.hub_repo = "valgur/monodepth2"

    def load_model(self):
        pretrained_model = "mono+stereo_640x192"
        encoder = torch.hub.load(self.hub_repo, "ResnetEncoder", pretrained_model, map_location=self.device)
        depth_decoder = torch.hub.load(self.hub_repo, "DepthDecoder", pretrained_model, map_location=self.device)
        encoder.to(self.device)
        depth_decoder.to(self.device)
        return depth_decoder, encoder

    @handle_alpha
    @torch.no_grad()
    def predict(self, input_image, colormap=None):
        h, w, d = input_image.shape
        assert d == 3, "Input image must be RGB"

        # LOADING PRETRAINED MODEL
        depth_decoder, encoder = self.model

        input_image = Image.fromarray(input_image)
        original_width, original_height = input_image.size
        input_image = input_image.resize((encoder.feed_width, encoder.feed_height), Image.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(self.device)

        # PREDICTION
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
        disp = disp.squeeze().cpu().numpy()
        disp /= disp.max()

        if colormap:
            out = apply_colormap(disp, colormap)
        else:
            out = to_rgb(disp)
        return (out * 255).astype(np.uint8)


model = Monodepth2()

if __name__ == '__main__':
    rpc_url = sys.argv[1]
    model.process_rpc(rpc_url)
