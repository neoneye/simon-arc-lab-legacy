import numpy as np
from PIL import Image

def convert_pilimage_to_onehot(pil_image: Image) -> np.ndarray:
    """
    Extract the Red channel from an RGB Image.

    Convert the Red channel from the range 0..255 to the range 0..12

    Encode the value as onehot.
    """

    # Get R channel data (ignoring G and B)
    r_channel_data = np.array(pil_image)[:,:,0]

    # Calculate the one-hot indices by dividing the red channel values into 13 bins
    one_hot_indices = np.minimum((r_channel_data // 21).astype(np.int32), 12)

    # Initialize a zeros array for one-hot encoding
    # Shape is (height, width, 13) to store 13 one-hot values for each pixel
    height, width = one_hot_indices.shape
    one_hot_array = np.zeros((height, width, 13), dtype=np.int32)

    # Perform one-hot encoding
    for i in range(13):
        mask = (one_hot_indices == i)
        one_hot_array[mask, i] = 1
    return one_hot_array
    