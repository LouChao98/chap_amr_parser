import random
import re
import string
from pathlib import Path


class ReservoirSampling:
    def __init__(self, k) -> None:
        self.reservoir = []
        self.k = k
        self.cnt = 0

    def consume(self, item):
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            j = random.randint(0, self.cnt)
            if j < self.k:
                item = re.sub(r" (\.\d+) ", r" 0\1 ", item)
                self.reservoir[j] = item
            self.cnt += 1


def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())


reservoir = ReservoirSampling(200_000)

for fpath in Path(
    "/mnt/research/lm1b/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled"
).glob("news.en-00*-of-00100"):
    print(fpath)

    for item in fpath.read_text().splitlines():
        length = len(item.split())
        if 5 <= length <= 150 and isascii(item) and not re.search(r" \.\d+[^\s]+ ", item):
            reservoir.consume(item)


with open("data/lm1b_200k.txt", "w") as f:
    for item in reservoir.reservoir:
        f.write(item)
        f.write("\n")
