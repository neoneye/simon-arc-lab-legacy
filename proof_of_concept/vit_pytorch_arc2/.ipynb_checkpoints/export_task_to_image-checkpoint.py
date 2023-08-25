import arc_json_model as ajm
import numpy as np
from simple_image import image_new, set_pixel, draw_rect, draw_box

# ARC images are maximum 30x30. With a 1px border around, then it's the IMAGE_SIZE 32x32.
IMAGE_SIZE = 32

# The color codes [0..9] are the ARC colors

# Color used when expanding the input/output images to size 30x30
COLOR_OUTSIDE = 10

# Color used for a 1px wide box around the input/output images
COLOR_PADDING = 11

# Color that indicates the single pixel that is to be predicted
# Color that indicates the box with the prediction areas
COLOR_HIGHLIGHT = 12

class ExportTaskToImage:
    def __init__(self, task: ajm.Task):
        self.task = task
        self.pixels = ExportTaskToImage.image_from_task(task)

    @classmethod
    def image_from_task(cls, task: ajm.Task) -> np.ndarray:
        # each ARC pair is IMAGE_SIZE wide
        # in total it's all the pairs * IMAGE_SIZE
        image_width = IMAGE_SIZE * len(task.pairs)

        # input images in the top row
        # output images in the bottom row
        # in total there are 2 rows, thus the image_height is IMAGE_SIZE * 2
        image_height = IMAGE_SIZE * 2
        
        pixels = image_new(image_width, image_height, COLOR_OUTSIDE)

        # copy input images
        for pair_index, pair in enumerate(task.pairs):
            for row_index, rows in enumerate(pair.input.pixels):
                for column_index, pixel in enumerate(rows):
                    x = IMAGE_SIZE * pair_index + 1 + column_index
                    y = IMAGE_SIZE * 0 + 1 + row_index
                    set_pixel(pixels, x, y, pixel)

        # copy output images
        for pair_index, pair in enumerate(task.pairs):
            for row_index, rows in enumerate(pair.output.pixels):
                for column_index, pixel in enumerate(rows):
                    x = IMAGE_SIZE * pair_index + 1 + column_index
                    y = IMAGE_SIZE * 1 + 1 + row_index
                    set_pixel(pixels, x, y, pixel)

        # draw boxes around images
        for pair_index, pair in enumerate(task.pairs):
            x = IMAGE_SIZE * pair_index
            draw_box(pixels, x, 0, IMAGE_SIZE, IMAGE_SIZE, COLOR_PADDING)
            draw_box(pixels, x, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, COLOR_PADDING)

        # mask out the test output areas
        # and draw a highlighted box around the test output areas
        for pair_index, pair in enumerate(task.pairs):
            if pair.pair_type != ajm.PairType.TEST:
                continue
            x = IMAGE_SIZE * pair_index
            width = pair.output.pixels.shape[1]
            height = pair.output.pixels.shape[0]
            draw_rect(pixels, x, IMAGE_SIZE, width + 2, height + 2, COLOR_OUTSIDE)
            draw_box(pixels, x, IMAGE_SIZE, width + 2, height + 2, COLOR_HIGHLIGHT)
        return pixels

    def image_with_mark(self, test_index: int, x: int, y: int) -> np.ndarray:
        pixels = self.pixels.copy()
        for pair_index, pair in enumerate(self.task.pairs):
            if pair.pair_type != ajm.PairType.TEST:
                continue
            if pair.pair_index != test_index:
                continue
            set_x = IMAGE_SIZE * pair_index + 1 + x
            set_y = IMAGE_SIZE + 1 + y
            set_pixel(pixels, set_x, set_y, COLOR_HIGHLIGHT)
        return pixels

if __name__ == '__main__':
    filename = 'testdata/af902bf9.json'
    task = ajm.Task.load(filename)
    #print(task)
    exporter = ExportTaskToImage(task)
    #print(exporter)
    pixels = exporter.image_with_mark(0, 3, 3)
    print("pixels.shape", pixels.shape)
    
