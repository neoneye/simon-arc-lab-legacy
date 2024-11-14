import arc_json_model as ajm
import matplotlib.pyplot as plt
from matplotlib import colors

def ajm_image_show(image: ajm.Image):
    """
    Install the function like this
    ajm.Image.show = ajm_image_show
    """
    
    cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25',
     '#282828', '#d0d0d0', '#FFFFFF'])
    norm = colors.Normalize(vmin=0, vmax=12)
    
    fig, axs = plt.subplots(1, 1, figsize=(5,5))
    axs.imshow(image.pixels, cmap=cmap, norm=norm)
    axs.set_title(image.id)
    axs.set_yticks(list(range(image.pixels.shape[0])))
    axs.set_xticks(list(range(image.pixels.shape[1])))
    plt.tight_layout()
    plt.show()
