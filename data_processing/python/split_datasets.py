from collections import Counter
from pathlib import Path

import pandas
from sklearn.model_selection import StratifiedKFold


# ===== AUX FUNCTIONS
def split_complete_pipeline(data_dir, split_dir, item_list, given_samples, which, aux_dir):
    # Per omic, get samples not included in common splits
    diff_dict = dict()
    miss_dict = dict()

    if which == 'omi':
        aux_pre = 'BRCA_'
        aux_suf = '_train.csv'
        sample_col = 'case_id'
    else:  # int
        aux_pre = ''
        aux_suf = '_4Q_prediction.tsv'
        sample_col = 'SAMPLE'

    print('\nDISTINCT SAMPLES:')
    for x in item_list:
        x_file = data_dir / f'{aux_pre}{x}{aux_suf}'
        all_samples = get_samples_in_omi(x_file, sample_col)

        diff_samples = list(set(all_samples) - set(given_samples))
        common_samples = list(set(all_samples).intersection(given_samples))
        miss_samples = list(set(given_samples) - set(all_samples))

        diff_dict[x] = diff_samples
        miss_dict[x] = miss_samples

        print(f'{x}: All ({len(all_samples)}), Diff ({len(diff_samples)}), Common ({len(common_samples)}), Miss ({len(miss_samples)})')

        # break

    # Split these remaining samples
    print('\nDATA SPLITS:')
    new_splits = dict()
    for x in item_list:
        diff_samples = diff_dict[x]
        clin_file = aux_dir / 'BRCA_CLIN_train.csv'
        new_splits[x] = get_split_samples(diff_samples, 5, clin_file)
        print(x, new_splits[x])

        # break

    # Get samples from common splits
    common_splits = dict()
    for i in range(5):
        split_file = split_dir / 'BENCHMARKING' / f'splits_{i}.csv'
        common_splits[i] = get_samples_in_split(split_file)

    # Merge NEW and COMMMON splits
    print('\nFINAL SPLITS:')
    for x in item_list:
        out_dir = split_dir / x.replace('_4Q', '')
        out_dir.mkdir(parents=True, exist_ok=True)
        miss_samples = miss_dict[x]

        for i in range(5):
            out_file = out_dir / f'splits_{i}.csv'
            merge_splits_and_save(common_splits[i], new_splits[x][i], miss_samples, out_file, f'{x}-{i}')

        # break

    # Check resulting sets
    print('\nCHECK RESULTS:')
    for x in item_list:
        print(x.replace('_4Q', ''))
        out_dir = split_dir / x.replace('_4Q', '')

        val_samples = list()
        for i in range(5):
            out_file = out_dir / f'splits_{i}.csv'
            split_data = get_samples_in_split(out_file)
            val_samples += split_data['val']

        x_file = data_dir / f'{aux_pre}{x}{aux_suf}'
        all_samples = get_samples_in_omi(x_file, sample_col)

        check_validation_set(val_samples, all_samples)


def get_samples_in_split(fpath: Path) -> dict:
    df = pandas.read_csv(fpath, sep=',', header=0, index_col=0)
    samples = {'train': df['train'].tolist(), 'val': [x for x in df['val'].tolist() if not pandas.isna(x)]}
    return samples


def get_samples_in_omi(fpath: Path, sample_col) -> list:
    df = pandas.read_csv(fpath, sep='\t', header=0, usecols=[sample_col])
    samples = df[sample_col].tolist()
    return samples


def get_split_samples(sample_list: list, n_splits: int, info_fpath: Path, group_cols: list = ['OUTCOME', 'IS_CENSORED'], sample_col: str = 'case_id') -> dict:
    if len(sample_list) > 0:
        # Initialize KFold with N splits (STRATIFIED)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=25)  # random state asegura siempre mismo split

        # Define input DF (considering outcome for balanced split)
        df_info = pandas.read_csv(info_fpath, sep='\t', header=0, usecols=[sample_col]+group_cols)
        df_split = df_info.copy()[df_info[sample_col].isin(sample_list)]
        df_split['GROUP'] = df_split.loc[:, group_cols].astype(str).agg('-'.join, axis=1)
        df_split = df_split.loc[:, [sample_col, 'GROUP']]

        # Get data splits
        output = dict()
        print('* Data splits:')
        for fold, (train_indices, val_indices) in enumerate(skf.split(df_split, df_split['GROUP'])):
            # Creates train/val groups from DF
            df_train = df_split.iloc[train_indices]
            df_val = df_split.iloc[val_indices]

            # Keep only train/val sample names
            train_set = df_train[sample_col].tolist()
            val_set = df_val[sample_col].tolist()

            print(f'  > FOLD {fold}: Train ({len(train_set)}) & Val ({len(val_set)})')

            # Return train and validation sets
            output[fold] = {'train': train_set, 'val': val_set}
    else:
        output = dict()
        for fold in range(n_splits):
            output[fold] = {'train': [], 'val': []}
        print('* Data splits:')

    return output


def merge_splits_and_save(set_1: dict, set_2: dict, to_drop: list, fpath: Path, x: str):
    # Merge train and validation sets (Drop samples missing in OMI data)
    train = [x for x in set_1['train'] if x not in to_drop] + set_2['train']
    val = [x for x in set_1['val'] if x not in to_drop] + set_2['val']

    # Store data in DF and export to given fpath
    df = pandas.DataFrame({'train': train,
                           'val': val + [None] * (len(train) - len(val))})
    df.to_csv(fpath, sep=',', index=True, header=True)

    print(x, len(train), len(val), fpath)


def check_validation_set(val_list: list, all_list: list):
    # Revisar si los elementos de validacion son unicos
    if len(val_list) != len(set(val_list)):
        print('* Duplicates in validation set:')

        counter = Counter(val_list)
        dups = [x for x, freq in counter.items() if freq > 1]
        print(dups)
    else:
        print('* NO duplicates in validation set')

    # Revisar si todas las muestras se han incluido en algun set de validacion
    if set(all_list).issubset(set(val_list)):
        print('* All samples in validation set')
    else:
        print('* Samples missing in validation set')
        print(set(all_list) - set(val_list))


# =========== MAIN
if __name__ == "__main__":
    # Define input files
    split_dir = Path('')
    omi_dir = Path('')
    int_dir = Path('')

    omi_list = ['CLIN', 'CNV', 'MIR', 'RNA', 'SNP']

    int_list = ['OMI', 'OMI_CLIN', 'OMI_RESNET', 'OMI_CLIN_RESNET', 'CLIN_RESNET']

    # Get samples included in BASE (commmon) splits
    split_file = split_dir / 'BENCHMARKING' / 'splits_0.csv'
    split_dict = get_samples_in_split(split_file)
    split_samples = split_dict['train'] + split_dict['val']
    print('COMMON SAMPLES:')
    print(len(split_samples), split_samples)

    # Get final splits per omic
    split_complete_pipeline(omi_dir, split_dir, omi_list, split_samples, 'omi', omi_dir)

    # Get final splits per integration file
    split_complete_pipeline(int_dir, split_dir, int_list, split_samples, 'int', omi_dir)


