import unittest
import arc_json_model as ajm
import export_task_to_image as etti

class TestExportTaskToImage(unittest.TestCase):
    def test_export(self):
        filename = 'testdata/af902bf9.json'
        task = ajm.Task.load(filename)
        exporter = etti.ExportTaskToImage(task)
        pixels = exporter.image_with_mark(0, 3, 3)
        self.assertEqual(pixels.shape, (64, 128))
