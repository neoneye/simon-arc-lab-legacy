from collections import Counter
import json
import heapq

class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_map):
    heap = [HuffmanNode(value, freq) for value, freq in freq_map.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def create_codes(node, prefix="", code_map={}):
    if node is not None:
        if node.value is not None:
            code_map[node.value] = prefix
        create_codes(node.left, prefix + "0", code_map)
        create_codes(node.right, prefix + "1", code_map)
    return code_map

def huffman_encode(data, code_map):
    return ''.join(code_map[item] for item in data)

def huffman_decode(encoded_data, root):
    decoded_output = []
    current_node = root
    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.value is not None:
            decoded_output.append(current_node.value)
            current_node = root

    return decoded_output

def huffman_tree_to_dict(node):
    if node is None:
        return None
    node_dict = {
        'value': node.value,
        'freq': node.freq,
        'left': huffman_tree_to_dict(node.left),
        'right': huffman_tree_to_dict(node.right)
    }
    return node_dict

def dict_to_huffman_tree(node_dict):
    if node_dict is None:
        return None
    node = HuffmanNode(node_dict['value'], node_dict['freq'])
    node.left = dict_to_huffman_tree(node_dict['left'])
    node.right = dict_to_huffman_tree(node_dict['right'])
    return node

def count_huffman_nodes(node):
    if node is None:
        return 0
    return 1 + count_huffman_nodes(node.left) + count_huffman_nodes(node.right)

def binary_string_to_bytes(bin_str):
    byte_array = bytearray()
    for i in range(0, len(bin_str), 8):
        byte_chunk = bin_str[i:i+8]
        byte_array.append(int(byte_chunk, 2))
    return bytes(byte_array)

if __name__ == '__main__':
    # Use the same huffman tree for all the images in the ARC task
    image_data0 = [1, 2, 3, 1, 1, 2, 3, 3, 3, 1]
    image_data1 = [5, 1, 2, 2, 1, 5, 3, 5, 3, 1]
    image_data2 = [5, 1, 2, 2, 1, 5, 3, 5, 3, 1]
    freq_map = Counter(image_data0)
    freq_map += Counter(image_data1)
    freq_map += Counter(image_data2)
    root = build_huffman_tree(freq_map)

    # Pretty print the Huffman tree
    huffman_tree_dict = huffman_tree_to_dict(root)
    json_huffman_tree = json.dumps(huffman_tree_dict, indent=4)
    print(json_huffman_tree)

    # Count the number of HuffmanNode objects in the tree
    node_count = count_huffman_nodes(root)
    print(f"Number of HuffmanNode objects in the tree: {node_count}")

    code_map = create_codes(root)
    print(f"Code Map: {code_map}")

    encoded_data = huffman_encode(image_data0, code_map)
    decoded_data = huffman_decode(encoded_data, root)
    print(f"Encoded Data: {encoded_data}")
    print(f"Decoded Data: {decoded_data}")
