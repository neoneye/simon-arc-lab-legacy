from game_of_life import game_of_life
from random_game_of_life import generate_random_game_of_life_string

input = generate_random_game_of_life_string(seed=43)
output = game_of_life(input)
print(input)
print('---')
print(output)
