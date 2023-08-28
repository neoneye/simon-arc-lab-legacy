import unittest
import arc_json_model as ajm

class TestArcJsonModel(unittest.TestCase):
    def test_load(self):
        filename = 'testdata/62c24649.json'
        task = ajm.Task.load(filename)
        self.assertEqual(len(task.pairs), 4)
        self.assertEqual(task.train_test(), (3, 1))
        pair0 = task.pairs[0]
        self.assertEqual(pair0.input.pixels.shape, (3, 3))
        self.assertEqual(pair0.output.pixels.shape, (6, 6))
