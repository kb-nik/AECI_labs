import sys
import os
import logging

# Логи для отладки
logging.basicConfig(
    filename='huffman.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class HuffNode:
    def __init__(self, symb=None, count=0, lft=None, rgt=None):
        self.symb = symb  
        self.count = count 
        self.lft = lft  # Лево
        self.rgt = rgt  # Право
class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, node):
        self.heap.append(node)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        min_node = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return min_node

    def _sift_up(self, idx):
        parent = (idx - 1) // 2
        while idx > 0 and (self.heap[idx].count < self.heap[parent].count or
                          (self.heap[idx].count == self.heap[parent].count and self.heap[idx].symb is not None and self.heap[parent].symb is not None and self.heap[idx].symb < self.heap[parent].symb)):
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            idx = parent
            parent = (idx - 1) // 2

    def _sift_down(self, idx):
        min_idx = idx
        size = len(self.heap)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            if left < size and (self.heap[left].count < self.heap[min_idx].count or
                               (self.heap[left].count == self.heap[min_idx].count and self.heap[left].symb is not None and self.heap[min_idx].symb is not None and self.heap[left].symb < self.heap[min_idx].symb)):
                min_idx = left
            if right < size and (self.heap[right].count < self.heap[min_idx].count or
                                (self.heap[right].count == self.heap[min_idx].count and self.heap[right].symb is not None and self.heap[min_idx].symb is not None and self.heap[right].symb < self.heap[min_idx].symb)):
                min_idx = right
            if min_idx == idx:
                break
            self.heap[idx], self.heap[min_idx] = self.heap[min_idx], self.heap[idx]
            idx = min_idx

    def __len__(self):
        return len(self.heap)


def count_freqs(data):
    freqs = {}
    for byte in data:
        freqs[byte] = freqs.get(byte, 0) + 1
    return freqs

# Дерево 
def make_huff_tree(freqs):
    if not freqs:
        return None
    heap = MinHeap()
    for symb, count in freqs.items():
        heap.push(HuffNode(symb, count))
    while len(heap) > 1:
        min1 = heap.pop()
        min2 = heap.pop()
        new_node = HuffNode(None, min1.count + min2.count, min1, min2)
        heap.push(new_node)
    return heap.pop()

# Код
def get_huff_codes(node, code='', codes_dict=None):
    if codes_dict is None:
        codes_dict = {}
    if node is None:
        return codes_dict
    if node.symb is not None:
        codes_dict[node.symb] = code or '0'
        return codes_dict
    get_huff_codes(node.lft, code + '0', codes_dict)
    get_huff_codes(node.rgt, code + '1', codes_dict)
    return codes_dict

# Дерево в биты
def tree_to_bits(node):
    if node is None:
        return ''
    if node.symb is not None:
        return '1' + f'{node.symb:08b}'
    return '0' + tree_to_bits(node.lft) + tree_to_bits(node.rgt)

# Из бит в дерево
def bits_to_tree(bit_str, idx=0):
    if idx >= len(bit_str):
        logging.error("Биты кончились")
        raise ValueError("Биты кончились")
    if bit_str[idx] == '1':
        if idx + 9 > len(bit_str):
            logging.error("Не хватает бит для символа")
            raise ValueError("Не хватает бит для символа")
        symb = int(bit_str[idx+1:idx+9], 2)
        return HuffNode(symb=symb), idx + 9
    idx += 1
    if idx >= len(bit_str):
        logging.error("Не хватает бит для узла")
        raise ValueError("Не хватает бит для узла")
    left_node, idx = bits_to_tree(bit_str, idx)
    if idx >= len(bit_str):
        logging.error("Правый потомок не найден")
        raise ValueError("Правый потомок не найден")
    right_node, idx = bits_to_tree(bit_str, idx)
    return HuffNode(lft=left_node, rgt=right_node), idx

# Кодирование
def encode_data(raw_bytes):
    if not raw_bytes:
        return b''
    freqs = count_freqs(raw_bytes)
    root = make_huff_tree(freqs)
    if root is None:
        return b''
    codes = get_huff_codes(root)
    print("Таблица кодов, зацени:")
    for symb, code in sorted(codes.items(), key=lambda x: len(x[1])):
        char = chr(symb) if 32 <= symb <= 126 else f'\\x{symb:02x}'
        print(f"'{char}' -> {code}")
        logging.info(f"Code: '{char}' -> {code}")
    bit_data = ''.join(codes[b] for b in raw_bytes)
    tree_bits = tree_to_bits(root)
    total_bits = len(tree_bits) + len(bit_data)
    pad_len = (8 - total_bits % 8) % 8
    all_bits = f'{pad_len:08b}' + tree_bits + bit_data + '0' * pad_len
    output = bytearray(b'ZHM!')
    for i in range(0, len(all_bits), 8):
        output.append(int(all_bits[i:i+8], 2))
    return bytes(output)

# Декодирование
def decode_data(comp_bytes):
    if len(comp_bytes) < 4:
        logging.error("Файл мелкий, нет ZHM!")
        raise ValueError("Файл мелкий, нет ZHM!")
    if comp_bytes[:4] != b'ZHM!':
        logging.error("Нет ZHM!")
        raise ValueError("Нет ZHM!")
    all_bits = ''.join(f'{b:08b}' for b in comp_bytes[4:])
    if len(all_bits) < 8:
        logging.error("Заголовок пустой")
        raise ValueError("Заголовок пустой")
    pad_len = int(all_bits[:8], 2)
    if len(all_bits) < 8 + pad_len:
        logging.error(f"Не хватает {pad_len} бит паддинга")
        raise ValueError(f"Не хватает {pad_len} бит паддинга")
    useful_bits = all_bits[8 : len(all_bits) - pad_len]
    if not useful_bits:
        return b''
    try:
        root, after_tree = bits_to_tree(useful_bits)
    except ValueError as e:
        logging.error(f"Ошибка в дереве: {e}")
        raise ValueError(f"Ошибка в дереве: {e}")
    decoded = bytearray()
    current = root
    for bit in useful_bits[after_tree:]:
        if current is None:
            logging.error("Current None")
            raise ValueError("Current None")
        current = current.lft if bit == '0' else current.rgt
        if current is None:
            logging.error("Провалился в None")
            raise ValueError("Провалился в None")
        if current.symb is not None:
            decoded.append(current.symb)
            current = root
    return bytes(decoded)

# Главная
def run_program():
    if len(sys.argv) != 4:
        print("Юзай: python3 main.py <mode> <input> <output>")
        print("mode: encode или decode")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    in_file = sys.argv[2]
    out_file = sys.argv[3]
    
    if not os.path.isfile(in_file):
        print(f"Файл {in_file} не нашел")
        sys.exit(1)
    
    if mode == 'decode' and not in_file.endswith('.zhm'):
        print("Только .zhm декодирую")
        sys.exit(1)
    
    if mode == 'encode' and not out_file.endswith('.zhm'):
        out_file = out_file + '.zhm'
        print(f"Сделал {out_file}")
    
    with open(in_file, 'rb') as f:
        data = f.read()
    
    if mode == 'encode':
        compressed = encode_data(data)
        with open(out_file, 'wb') as f:
            f.write(compressed)
        orig_size = len(data)
        comp_size = len(compressed)
        if orig_size > 0:
            percent = (orig_size - comp_size) / orig_size * 100
            print(f"Сжал {orig_size} байт до {comp_size} байт ({percent:.2f}% экономии)")
            logging.info(f"Encode: {in_file} ({orig_size}B) -> {out_file} ({comp_size}B), {percent:.2f}% saved")
        else:
            print("Пустой файл")
            logging.info("Encode: empty file")
        if comp_size >= orig_size:
            print("Мелкие файлы не жмутся из-за дерева")
    
    elif mode == 'decode':
        try:
            decompressed = decode_data(data)
            with open(out_file, 'wb') as f:
                f.write(decompressed)
            print(f"Распаковал {len(data)} байт до {len(decompressed)} байт")
            logging.info(f"Decode: {in_file} ({len(data)}B) -> {out_file} ({len(decompressed)}B)")
            # Проверка совпадения
            if os.path.isfile(in_file.replace('.zhm', '.txt')):
                with open(in_file.replace('.zhm', '.txt'), 'rb') as f_orig:
                    orig_data = f_orig.read()
                if orig_data == decompressed:
                    print("Совпадает с оригиналом")
                else:
                    print("Не совпадает")
        except ValueError as e:
            print(f"Ошибка: {e}")
            logging.error(f"Decode error: {e}")
            sys.exit(1)
    
    else:
        print("Режим не тот")
        sys.exit(1)

if __name__ == '__main__':
    run_program()
