import random

def generate_random_byte_array(length, seed):
    random.seed(seed)
    return bytearray(random.randint(0, 255) for _ in range(length))

if __name__ == "__main__":
    length = 10
    seed = 42
    byte_array = generate_random_byte_array(length, seed)
    print(byte_array)
