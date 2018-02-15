import os
import pandas as pd


LYRICS_FILE = './songlyrics.zip'


if not os.path.exists('lyrics'):
    os.mkdir('lyrics')

for rec in pd.read_csv(LYRICS_FILE).itertuples():
    name = '%s-%s' % (rec.artist.lower(), rec.song.lower().replace(' ', '-'))
    with open(os.path.join('lyrics', '%s.txt' % name), 'w') as f:
        # remove trailing spaces
        text = '\n'.join([L.strip() for L in rec.text.split('\n')])
        f.write(text)
