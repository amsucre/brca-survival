from pathlib import Path

import pandas


# ===== AUX FUNCTIONS
def split_str(x):
    if x['FULL_NAME']:
        return x['FULL_NAME'].rstrip(')').split(' (')
    else:
        return ['', '']


# =========== MAIN
if __name__ == "__main__":
    # Read oncotree file and extract codes, names and levels
    # # Define input path
    parent_dir = Path(__file__).resolve().parent.parent  # gets "data_processing" path
    fpath = parent_dir / 'data' / 'oncotree_tumor_types.txt'

    # # Read data into DF
    level_cols = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5', 'level_6', 'level_7']
    df = pandas.read_csv(fpath, sep='\t', header=0, usecols=level_cols)

    # # Keep only BREAST codes
    df = df.loc[df['level_1'] == 'Breast (BREAST)', :]

    # # Manage NaN (drop Nan-columns and replace with empty str)
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna('', inplace=True)
    print(df)

    # # Get list of unique values per column
    sets_per_level = dict()
    for lvl in df.columns.tolist():
        sets_per_level[lvl[-1]] = list(set(df[lvl].tolist()))  # save list per level. drop duplicates

    # # Define tuples having key, string
    rows = [(value, lvl) for lvl, values in sets_per_level.items() for value in values if value]

    # # Define new dataframe
    df = pandas.DataFrame(rows, columns=['FULL_NAME', 'LEVEL'])
    df[['NAME', 'CODE']] = df.apply(split_str, result_type='expand', axis=1)

    # # Drop full name and rearrange columns
    df = df.loc[:, ['NAME', 'CODE', 'LEVEL']]

    # # Save data
    outpath = parent_dir / 'data' / 'oncotree_CODES.txt'
    df.to_csv(outpath, sep='\t', index=False)
