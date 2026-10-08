"""Optional reproducible input schedules; hashes do not lock or encrypt state."""
from __future__ import annotations
import hashlib
import unicodedata

KEYPHRASES = [
    'donald trump tower', 'ing amsterdamro', 'abn', 'bani de banne',
    'chef', 'buikslotermeerplein', 'andere kant van het ij', 'winkelcentrumen',
]

def phrase_digest(text):
    if not isinstance(text, str) or not text:
        raise ValueError('Seed phrase must be a nonempty string')
    return hashlib.sha256(unicodedata.normalize('NFC', text).encode('utf-8')).hexdigest()

def mutation_bit(digest, chain_id, tick):
    if digest is None:
        return (tick + 1) % 2 == 0
    raw = bytes.fromhex(digest)
    if len(raw) != 32:
        raise ValueError('Seed digest must contain 32 bytes')
    bit = (tick + chain_id) % 256
    return (raw[bit // 8] >> (bit % 8)) & 1
