import random
import os
import shutil


files = os.listdir('lyrics')

def split():
    r = random.random()
    if 0 <= r <=0.33:
        return 0
    elif 0.3 < r <= 0.66:
        return 1
    return 2
splits = [(f, split()) for f in files]

train = [f for f, s in splits if s == 0]
test = [f for f, s in splits if s == 1]
holdout = [f for f, s in splits if s == 2]

for f in train:
    shutil.copyfile(f, os.path.join('lyrics-train', os.path.basename(f)))
for f in test:
    shutil.copyfile(f, os.path.join('lyrics-test', os.path.basename(f)))
for f in holdout:
    shutil.copyfile(f, os.path.join('lyrics-holdout', os.path.basename(f)))
