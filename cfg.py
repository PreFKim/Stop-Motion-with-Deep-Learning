import models
from losses import Focal_IoU

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 320

num_classes = 1

unet_level = 5

initial_features = 32

model = models.layerUNET(
    n_levels=unet_level,
    DSV=True,
    initial_features=initial_features,
    IMAGE_HEIGHT=IMAGE_HEIGHT,
    IMAGE_WIDTH=IMAGE_WIDTH,
    out_channels=num_classes
    )

loss = Focal_IoU
