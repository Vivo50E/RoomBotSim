#!/usr/bin/env python3
"""Generate the LAN handoff QR. Standard library plus Pillow, which the project already requires.

Usage: python static/generate_qr.py [http://LAN-IP:PORT/]

An earlier version imported cv2 to do the encoding. OpenCV is not in requirements.txt and is not in the
project venv, so the script only ran under an ambient interpreter that happened to have it. The byte-mode
encoder below removes that dependency: level M, versions 1-10, which covers any LAN URL.
"""
import os, socket, sys
from PIL import Image


def lan_ip():
    # UDP connect asks the OS for the outward interface; it sends no packets.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# --- Galois field GF(256) for Reed-Solomon
EXP = [0] * 512
LOG = [0] * 256
_x = 1
for _i in range(255):
    EXP[_i] = _x
    LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    EXP[_i] = EXP[_i - 255]


def _mul(a, b):
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]


def _rs_generator(n):
    g = [1]
    for i in range(n):
        g2 = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            g2[j] ^= c
            g2[j + 1] ^= _mul(c, EXP[i])
        g = g2
    return g


def _rs_encode(data, n):
    gen = _rs_generator(n)
    res = list(data) + [0] * n
    for i in range(len(data)):
        c = res[i]
        if c:
            for j, gc in enumerate(gen):
                res[i + j] ^= _mul(gc, c)
    return res[len(data):]


# version -> (total codewords, ec codewords per block, [blocks in group1, blocks in group2])
# level M only
VERSIONS = {
    1: (26, 10, [(1, 16)]),
    2: (44, 16, [(1, 28)]),
    3: (70, 26, [(1, 44)]),
    4: (100, 18, [(2, 32)]),
    5: (134, 24, [(2, 43)]),
    6: (172, 16, [(4, 27)]),
    7: (196, 18, [(4, 31)]),
    8: (242, 22, [(2, 38), (2, 39)]),
    9: (292, 22, [(3, 36), (2, 37)]),
    10: (346, 26, [(4, 43), (1, 44)]),
}
ALIGN = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
         7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50]}
# format info bits for level M (0b00) and masks 0-7, already XORed with 0x5412
FORMAT_M = [0x5412, 0x5125, 0x5E7C, 0x5B4B, 0x45F9, 0x40CE, 0x4F97, 0x4AA0]


def _capacity(v):
    _, ecw, groups = VERSIONS[v]
    return sum(n * d for n, d in groups)


def _pick_version(nbytes):
    for v in range(1, 11):
        # 4 bits mode + 8 or 16 bits length + data
        lenbits = 8 if v < 10 else 16
        if (4 + lenbits + nbytes * 8 + 7) // 8 <= _capacity(v):
            return v
    raise ValueError("payload too long for version 10")


def _bitstream(data, v):
    lenbits = 8 if v < 10 else 16
    bits = []
    def put(val, n):
        for i in range(n - 1, -1, -1):
            bits.append((val >> i) & 1)
    put(0b0100, 4)               # byte mode
    put(len(data), lenbits)
    for b in data:
        put(b, 8)
    cap = _capacity(v) * 8
    put(0, min(4, cap - len(bits)))                 # terminator
    while len(bits) % 8:
        bits.append(0)
    pad = [0xEC, 0x11]
    i = 0
    while len(bits) // 8 < _capacity(v):
        put(pad[i % 2], 8)
        i += 1
    return [int("".join(str(b) for b in bits[k:k + 8]), 2) for k in range(0, len(bits), 8)]


def _interleave(codewords, v):
    _, ecw, groups = VERSIONS[v]
    blocks, pos = [], 0
    for count, dsize in groups:
        for _ in range(count):
            blocks.append(codewords[pos:pos + dsize])
            pos += dsize
    ecs = [_rs_encode(b, ecw) for b in blocks]
    out = []
    for i in range(max(len(b) for b in blocks)):
        for b in blocks:
            if i < len(b):
                out.append(b[i])
    for i in range(ecw):
        for e in ecs:
            out.append(e[i])
    return out


def _matrix(v):
    size = 17 + 4 * v
    m = [[None] * size for _ in range(size)]

    def finder(r, c):
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size:
                    on = (0 <= dr <= 6 and 0 <= dc <= 6 and
                          (dr in (0, 6) or dc in (0, 6) or (2 <= dr <= 4 and 2 <= dc <= 4)))
                    m[rr][cc] = 1 if on else 0
    finder(0, 0); finder(0, size - 7); finder(size - 7, 0)
    for i in range(8, size - 8):                      # timing
        b = 1 if i % 2 == 0 else 0
        if m[6][i] is None: m[6][i] = b
        if m[i][6] is None: m[i][6] = b
    for r in ALIGN[v]:                                 # alignment
        for c in ALIGN[v]:
            if (r < 8 and c < 8) or (r < 8 and c > size - 9) or (r > size - 9 and c < 8):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    m[r + dr][c + dc] = 1 if (abs(dr) == 2 or abs(dc) == 2 or (dr == 0 and dc == 0)) else 0
    m[size - 8][8] = 1                                 # dark module
    for i in range(9):                                 # reserve format areas
        if m[8][i] is None: m[8][i] = 0
        if m[i][8] is None: m[i][8] = 0
    for i in range(8):
        if m[8][size - 1 - i] is None: m[8][size - 1 - i] = 0
        if m[size - 1 - i][8] is None: m[size - 1 - i][8] = 0
    return m, size


def _place(m, size, data_bits):
    idx = 0
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for r in rows:
            for c in (col, col - 1):
                if m[r][c] is None:
                    m[r][c] = data_bits[idx] if idx < len(data_bits) else 0
                    m[r][c] |= 0  # keep int
                    idx += 1
        upward = not upward
        col -= 2
    return m


MASKS = [lambda r, c: (r + c) % 2 == 0,
         lambda r, c: r % 2 == 0,
         lambda r, c: c % 3 == 0,
         lambda r, c: (r + c) % 3 == 0,
         lambda r, c: (r // 2 + c // 3) % 2 == 0,
         lambda r, c: (r * c) % 2 + (r * c) % 3 == 0,
         lambda r, c: ((r * c) % 2 + (r * c) % 3) % 2 == 0,
         lambda r, c: ((r + c) % 2 + (r * c) % 3) % 2 == 0]


def encode(text, mask=2):
    data = text.encode("utf-8")
    v = _pick_version(len(data))
    cws = _interleave(_bitstream(data, v), v)
    bits = []
    for b in cws:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    base, size = _matrix(v)
    reserved = [[base[r][c] is not None for c in range(size)] for r in range(size)]
    m = _place([row[:] for row in base], size, bits)
    for r in range(size):
        for c in range(size):
            if not reserved[r][c] and MASKS[mask](r, c):
                m[r][c] ^= 1
    fmt = FORMAT_M[mask]
    for i in range(15):
        bit = (fmt >> (14 - i)) & 1
        if i < 6:      m[8][i] = bit
        elif i == 6:   m[8][7] = bit
        elif i == 7:   m[8][8] = bit
        elif i == 8:   m[7][8] = bit
        else:          m[14 - i][8] = bit
        if i < 8:      m[size - 1 - i][8] = bit
        else:          m[8][size - 15 + i] = bit
    m[size - 8][8] = 1
    return m, size


def write_png(text, path, scale=8, quiet=4):
    """Four white modules of quiet zone, then nearest-neighbour scaling to crisp pixels."""
    m, size = encode(text)
    side = (size + 2 * quiet) * scale
    img = Image.new("L", (side, side), 255)
    px = img.load()
    for r in range(size):
        for c in range(size):
            if m[r][c]:
                y0, x0 = (r + quiet) * scale, (c + quiet) * scale
                for dy in range(scale):
                    for dx in range(scale):
                        px[x0 + dx, y0 + dy] = 0
    img.save(path)
    return path


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://%s:%s/" % (lan_ip(), os.getenv("WALKIN_PORT", "8000"))
    if "127.0.0.1" in target or "localhost" in target:
        raise SystemExit("Refusing a loopback QR; pass a reachable LAN URL explicitly.")
    write_png(target, os.path.join(os.path.dirname(os.path.abspath(__file__)), "qr.png"))
    print(target)
