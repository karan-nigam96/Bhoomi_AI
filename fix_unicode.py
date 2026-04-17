# fix_unicode.py — strips non-cp1252 chars from scripts so Windows console can run them
import os

FILES = [
    'dataset/build_authentic_dataset.py',
    'train_v2.py',
]

REPLACEMENTS = [
    ('\u2500', '-'), ('\u2502', '|'), ('\u251c', '+'), ('\u2514', '+'),
    ('\u253c', '+'), ('\u2588', '#'), ('\u2019', "'"), ('\u2014', '--'),
    ('\u2013', '-'), ('\u2713', '[OK]'), ('\u274c', '[FAIL]'),
    ('\u26a0', '[!]'), ('\u2705', '[OK]'), ('\u2015', '-'),
    ('\u00b0', ' deg'), ('\u2714', '[OK]'), ('\u00d7', 'x'),
    ('\u2212', '-'), ('\u00b1', '+/-'),
]

for fpath in FILES:
    if not os.path.exists(fpath):
        print(f'SKIP (not found): {fpath}')
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    original_len = len(content)
    for bad, good in REPLACEMENTS:
        content = content.replace(bad, good)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Cleaned: {fpath}  ({original_len} -> {len(content)} chars)')

print('All done.')
