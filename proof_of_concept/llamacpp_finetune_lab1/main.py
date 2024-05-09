import my_arc_thing
import my_arc_thing.arc_json_model as ajm
import os

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "my_arc_thing", "assets", "68b67ca3.json")
task = ajm.Task.load(path)
print(task)
