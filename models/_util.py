def to_rgb(image):
    if len(image.shape) == 2:
        image = image[:, :, None]
    return image[:, :, (0, 0, 0)]


def apply_colormap(image, cmap='magma'):
    # image must be in 0-1 range
    import matplotlib.cm as cm
    mapper = cm.ScalarMappable(norm=lambda x: x, cmap=cmap)
    return mapper.to_rgba(image)[:, :, :3]
