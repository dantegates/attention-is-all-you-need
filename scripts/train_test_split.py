import random
import os
import shutil


dir_ = 'songnames'
files = [os.path.join(dir_, f) for f in os.listdir(dir_)]

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

train_dir = os.path.join('%s-train' % dir_)
test_dir = os.path.join('%s-test' % dir_)
holdout_dir = os.path.join('%s-holdout' % dir_)

for d in [train_dir, test_dir, holdout_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

for f in train:
    shutil.copyfile(f, os.path.join(train_dir, os.path.basename(f)))
for f in test:
    shutil.copyfile(f, os.path.join(test_dir, os.path.basename(f)))
for f in holdout:
    shutil.copyfile(f, os.path.join(holdout_dir, os.path.basename(f)))
