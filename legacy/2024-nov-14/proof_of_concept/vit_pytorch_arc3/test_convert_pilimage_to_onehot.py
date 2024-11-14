import unittest
import numpy as np
from PIL import Image
from convert_pilimage_to_onehot import convert_pilimage_to_onehot

class TestConvertPILImageToOneHot(unittest.TestCase):
    def test_convert(self):
        width = 2
        height = 7
        image = Image.new('RGB', (width, height))
        image.putpixel((0,0), (0,0,0))
        image.putpixel((1,0), (21,0,0))
        image.putpixel((0,1), (42,0,0))
        image.putpixel((1,1), (63,0,0))
        image.putpixel((0,2), (85,0,0))
        image.putpixel((1,2), (106,0,0))
        image.putpixel((0,3), (127,0,0))
        image.putpixel((1,3), (148,0,0))
        image.putpixel((0,4), (170,0,0))
        image.putpixel((1,4), (191,0,0))
        image.putpixel((0,5), (212,0,0))
        image.putpixel((1,5), (233,0,0))
        image.putpixel((0,6), (255,0,0))
        image.putpixel((1,6), (255,0,0))
        #image.show()
        
        one_hot_array = convert_pilimage_to_onehot(image)

        self.assertEqual(one_hot_array.shape, (7, 2, 13))
        self.assertEqual(one_hot_array[0][0].argmax(), 0)
        self.assertEqual(one_hot_array[0][1].argmax(), 1)
        self.assertEqual(one_hot_array[1][0].argmax(), 2)
        self.assertEqual(one_hot_array[1][1].argmax(), 3)
        self.assertEqual(one_hot_array[5][0].argmax(), 10)
        self.assertEqual(one_hot_array[5][1].argmax(), 11)
        self.assertEqual(one_hot_array[6][0].argmax(), 12)
        self.assertEqual(one_hot_array[6][1].argmax(), 12)
