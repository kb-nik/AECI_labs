import sys
import os
import time
import logging

# Логи для отладки
logging.basicConfig(
    filename='arithmetic.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Константы для арифметического кодирования (целочисленная реализация)
# Используем 65535 как в книге Ватолина (стр. 39)
DELITEL = 65535
FIRST_QTR = 16384  # DELITEL // 4
HALF = 32768  # DELITEL // 2
THIRD_QTR = 49152  # FIRST_QTR * 3

class FreqTable:
    def __init__(self, data):
        self.freqs = {}
        for b in data:
            self.freqs[b] = self.freqs.get(b, 0) + 1
        if not self.freqs:
            raise ValueError("Пустые данные")
        self.symbols = sorted(self.freqs.keys())
        self.b = [0]
        cum = 0
        for s in self.symbols:
            cum += self.freqs[s]
            self.b.append(cum)
        self.total = cum
        self.sym_to_idx = {s: i for i, s in enumerate(self.symbols)}
    
    def get_range(self, symbol):
        idx = self.sym_to_idx[symbol]
        return self.b[idx], self.b[idx + 1]
    
    def find_symbol(self, value):
        left, right = 0, len(self.symbols) - 1
        while left <= right:
            mid = (left + right) // 2
            low = self.b[mid]
            high = self.b[mid + 1]
            if value < low:
                right = mid - 1
            elif value >= high:
                left = mid + 1
            else:
                return self.symbols[mid]
        raise ValueError(f"Символ не найден: value={value}, total={self.total}")
    
    def serialize(self):
        out = len(self.symbols).to_bytes(2, 'little')
        for s in self.symbols:
            freq = self.freqs[s]
            out += bytes([s]) + freq.to_bytes(4, 'little')
        return out
    
    @staticmethod
    def deserialize(data, offset):
        if offset + 2 > len(data):
            raise ValueError("Нет данных")
        n = int.from_bytes(data[offset:offset+2], 'little')
        offset += 2
        freqs = {}
        for _ in range(n):
            if offset + 5 > len(data):
                raise ValueError("Таблица обрезана")
            symbol = data[offset]
            freq = int.from_bytes(data[offset+1:offset+5], 'little')
            freqs[symbol] = freq
            offset += 5
        table = FreqTable.__new__(FreqTable)
        table.freqs = freqs
        table.symbols = sorted(freqs.keys())
        table.b = [0]
        cum = 0
        for s in table.symbols:
            cum += freqs[s]
            table.b.append(cum)
        table.total = cum
        table.sym_to_idx = {s: i for i, s in enumerate(table.symbols)}
        return table, offset

class BitWriter:
    def __init__(self):
        self.bits = []
    
    def write(self, bit):
        self.bits.append(1 if bit else 0)
    
    def to_bytes(self):
        result = bytearray()
        current = 0
        bit_count = 0
        for bit in self.bits:
            current = (current << 1) | bit
            bit_count += 1
            if bit_count == 8:
                result.append(current)
                current = 0
                bit_count = 0
        if bit_count > 0:
            current <<= (8 - bit_count)
            result.append(current)
        valid_bits = bit_count if bit_count > 0 else 8
        result.insert(0, valid_bits)
        return bytes(result)

class BitReader:
    def __init__(self, data):
        if not data:
            raise ValueError("Пустой поток")
        self.valid_bits_last = data[0]
        if not (1 <= self.valid_bits_last <= 8):
            raise ValueError("Некорректные биты")
        self.bytes = data[1:]
        self.byte_idx = 0
        self.bit_idx = 0
    
    def read(self):
        if self.byte_idx >= len(self.bytes):
            return 0
        valid_bits = 8
        if self.byte_idx == len(self.bytes) - 1:
            valid_bits = self.valid_bits_last
        if self.bit_idx >= valid_bits:
            self.byte_idx += 1
            self.bit_idx = 0
            if self.byte_idx >= len(self.bytes):
                return 0
            valid_bits = 8
            if self.byte_idx == len(self.bytes) - 1:
                valid_bits = self.valid_bits_last
        byte_val = self.bytes[self.byte_idx]
        bit = (byte_val >> (7 - self.bit_idx)) & 1
        self.bit_idx += 1
        return bit

def encode_data(raw_data):
    if not raw_data:
        logging.info("Encode: empty data")
        return b''
    table = FreqTable(raw_data)
    logging.info(f"Encode: built frequency table with {len(table.symbols)} symbols, total={table.total}")
    li = 0
    hi = DELITEL - 1
    bits_to_follow = 0
    writer = BitWriter()
    for symbol in raw_data:
        a_c, b_c = table.get_range(symbol)
        interval_len = hi - li + 1
        hi = li + (b_c * interval_len) // table.total - 1
        li = li + (a_c * interval_len) // table.total
        while True:
            if hi < HALF:
                writer.write(0)
                for _ in range(bits_to_follow):
                    writer.write(1)
                bits_to_follow = 0
            elif li >= HALF:
                writer.write(1)
                for _ in range(bits_to_follow):
                    writer.write(0)
                bits_to_follow = 0
                li -= HALF
                hi -= HALF
            elif li >= FIRST_QTR and hi < THIRD_QTR:
                bits_to_follow += 1
                li -= FIRST_QTR
                hi -= FIRST_QTR
            else:
                break
            li = li * 2
            hi = hi * 2 + 1
    bits_to_follow += 1
    if li < FIRST_QTR:
        writer.write(0)
        for _ in range(bits_to_follow):
            writer.write(1)
    else:
        writer.write(1)
        for _ in range(bits_to_follow):
            writer.write(0)
    result = table.serialize() + len(raw_data).to_bytes(8, 'little') + writer.to_bytes()
    logging.info(f"Encode: output {len(result)} bytes for {len(raw_data)} input bytes")
    return result

def decode_data(comp_data):
    offset = 0
    table, offset = FreqTable.deserialize(comp_data, offset)
    logging.info(f"Decode: loaded frequency table with {len(table.symbols)} symbols")
    if offset + 8 > len(comp_data):
        logging.error("Decode: missing data length")
        raise ValueError("Нет длины")
    orig_len = int.from_bytes(comp_data[offset:offset+8], 'little')
    offset += 8
    logging.info(f"Decode: original length = {orig_len} bytes")
    if orig_len == 0:
        return b''
    bitstream = comp_data[offset:]
    reader = BitReader(bitstream)
    li = 0
    hi = DELITEL - 1
    value = 0
    for _ in range(16):
        value = (value << 1) | reader.read()
    result = bytearray()
    for _ in range(orig_len):
        interval_len = hi - li + 1
        # Вычисляем частоту - формула из книги (стр. 41)
        freq = (value - li) * table.total // interval_len  
        # Важная проверка для последнего символа в интервале (стр. 41 Ватолина)
        if freq >= table.total:
            freq = table.total - 1
        symbol = table.find_symbol(freq)
        result.append(symbol)
        a_c, b_c = table.get_range(symbol)
        # Точно такие же формулы как в кодере
        hi = li + (b_c * interval_len) // table.total - 1
        li = li + (a_c * interval_len) // table.total
        while True:
            if hi < HALF:
                pass
            elif li >= HALF:
                li -= HALF
                hi -= HALF
                value -= HALF
            elif li >= FIRST_QTR and hi < THIRD_QTR:
                li -= FIRST_QTR
                hi -= FIRST_QTR
                value -= FIRST_QTR
            else:
                break
            li = li * 2
            hi = hi * 2 + 1
            value = (value * 2) | reader.read()
    logging.info(f"Decode: successfully decoded {len(result)} bytes")
    return bytes(result)

def run_program():
    if len(sys.argv) != 4:
        print("Юзай: python3 main.py <mode> <input> <output>")
        print("mode: encode или decode")
        sys.exit(1)
    mode = sys.argv[1].lower()
    in_file = sys.argv[2]
    out_file = sys.argv[3]
    if not os.path.isfile(in_file):
        print(f"Файл {in_file} не найден")
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
        t_start = time.time()
        compressed = encode_data(data)
        t_end = time.time()
        with open(out_file, 'wb') as f:
            f.write(compressed)
        orig_size = len(data)
        comp_size = len(compressed)
        if orig_size > 0:
            ratio = orig_size / comp_size if comp_size > 0 else float('inf')
            percent = (orig_size - comp_size) / orig_size * 100
            print(f"Сжал {orig_size} байт до {comp_size} байт")
            print(f"Степень сжатия: {ratio:.3f} ({percent:.2f}% экономии)")
            print(f"Время: {t_end - t_start:.4f} сек")
            logging.info(f"Encode: {in_file} ({orig_size}B) -> {out_file} ({comp_size}B), {percent:.2f}% saved, time={t_end-t_start:.4f}s")
        else:
            print("Пустой файл")
            logging.info("Encode: empty file")
        if comp_size >= orig_size:
            print("Маленькие файлы не жмутся из-за таблицы частот")
    elif mode == 'decode':
        try:
            t_start = time.time()
            decompressed = decode_data(data)
            t_end = time.time()
            with open(out_file, 'wb') as f:
                f.write(decompressed)
            print(f"Распаковал {len(data)} байт до {len(decompressed)} байт")
            print(f"Время: {t_end - t_start:.4f} сек")
            logging.info(f"Decode: {in_file} ({len(data)}B) -> {out_file} ({len(decompressed)}B), time={t_end-t_start:.4f}s")
            orig_path = in_file.replace('.zhm', '.txt')
            if os.path.isfile(orig_path):
                with open(orig_path, 'rb') as f_orig:
                    orig_data = f_orig.read()
                if orig_data == decompressed:
                    print("Совпадает с оригиналом")
                    logging.info("Decode: matches original file")
                else:
                    print("Не совпадает с оригиналом")
                    logging.error("Decode: does NOT match original file")
        except Exception as e:
            print(f"Ошибка: {e}")
            logging.error(f"Decode error: {e}")
            sys.exit(1)
    else:
        print("Режим не тот")
        sys.exit(1)

if __name__ == '__main__':
    run_program()
