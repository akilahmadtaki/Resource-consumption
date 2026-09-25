from datasets import load_dataset

for n in ['three', 'five', 'seven']:
    subset = f'logical_deduction_{n}_objects'
    d = load_dataset('lukaemon/bbh', subset)['test']
    print('=' * 70)
    print(f'{subset}  —  {len(d)} rows,  fields: {d.column_names}')
    print('=' * 70)
    print(d[0]['input'])
    print()
    print('TARGET:', d[0]['target'])
    print()
