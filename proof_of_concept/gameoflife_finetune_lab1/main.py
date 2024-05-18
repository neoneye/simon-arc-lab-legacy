from game_of_life import GameOfLife
from random_game_of_life import generate_random_game_of_life_string
from game_of_life_mutator import GameOfLifeMutator

input = generate_random_game_of_life_string(seed=43)
output = GameOfLife.create(input, wrap_x=False, wrap_y=False, iterations=1).output_str
print(input)
print('---')
print(output)
print('---')

mutator = GameOfLifeMutator()
seed = 0
mutated_input = mutator.mutate(input, seed=seed)
mutated_output = mutator.mutate(output, seed=seed)

print(mutated_input)
print('---')
print(mutated_output)
