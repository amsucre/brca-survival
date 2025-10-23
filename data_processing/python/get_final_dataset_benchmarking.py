import os
import random

import pandas
from sklearn.model_selection import train_test_split, StratifiedKFold


# ===== AUX FUNCTIONS
def get_feature_df(inpath: str, auxpath: str):
    # Read main DF - Only relevant columns (index,case_id,oncotree_code,survival_months,censorship)
    df = pandas.read_csv(inpath, header=0, index_col=0, usecols=[0, 1, 6, 7], sep=',')
    df.drop_duplicates(inplace=True)
    print(f'* Input DF > {len(df)} (Unique IDs)')

    # Split survival time in 4 quartiles
    df['survival_group'] = pandas.qcut(df['survival_months'], q=4, labels=False)

    # Read extra clinical data to get molecular subtype (PAM50)
    df_aux = pandas.read_csv(auxpath, header=0, usecols=['sampleID', 'PAM50Call_RNAseq', 'PAM50_mRNA_nature2012'], sep='\t')
    df_aux = df_aux[df_aux['sampleID'].astype(str).str.endswith('01')]  # Drop normal samples >>> 11
    df_aux['case_id'] = df_aux['sampleID'].apply(lambda x: '-'.join(x.split('-')[0:3]))  # Get case ID
    df_aux['pam50'] = df_aux.loc[:, ['PAM50Call_RNAseq', 'PAM50_mRNA_nature2012']].apply(get_pam50, axis=1)  # Get PAM50
    df_aux.drop(['sampleID', 'PAM50Call_RNAseq', 'PAM50_mRNA_nature2012'], inplace=True, axis=1)
    df_aux['pam50'] = df_aux['pam50'].fillna('unknown')
    df_aux = df_aux.groupby('case_id', as_index=False)['pam50'].apply(select_pam)

    # Merge data with aux to get PAM50 info
    df = df.merge(df_aux, on='case_id', how='left')
    df['pam50'] = df['pam50'].fillna('unknown')

    # Define GROUP for data splitting
    df['GROUP'] = df.loc[:, ['survival_group', 'censorship', 'pam50']].astype(str).agg('-'.join, axis=1)

    return df.loc[:, ['case_id', 'GROUP']]


def get_pam50(row):
    matches = {'Basal-like': 'Basal', 'Her2-enriched': 'Her2', 'Luminal A': 'LumA', 'Luminal B': 'LumB', 'Normal-like': 'Normal'}
    if row['PAM50Call_RNAseq']:  # Default. Most recent (RNA-Seq)
        return row['PAM50Call_RNAseq']

    if row['PAM50_mRNA_nature2012']:  # Alternative. Older (Microarray)
        return matches.get(row['PAM50_mRNA_nature2012'], 'unknown')

    return 'unknown'  # No classification found


def select_pam(group):
    group = set(group)
    if len(group) == 1:
        return group.pop()  # All the same. Keep 1
    elif 'unknown' in group and len(group) == 2:
        return next(v for v in group if v != 'unknown')  # Something + Unknown > Keep unknown
    else:
        print(f'Revisar: {list(group)}')  # Otherwise > Print
        return group.pop()


def get_sample_from_img_name(inpath: str, outpath: str):
    df = pandas.read_csv(inpath, header=None, names=['FILENAME'])

    df['SAMPLE'] = df['FILENAME'].apply(lambda x: '-'.join(x.split('-')[0:3]))
    print(f'* Input IMG DF size: {df.shape}')

    df.to_csv(outpath, sep='\t', index=False)

    return list(set(df['SAMPLE'].tolist()))


def get_random_set(sample_list: list, n_all: int, pct: int):
    n_get = round(n_all * pct / 100)
    random.seed(25)
    chosen = random.sample(sample_list, n_get)

    return chosen


def split_train_test(ids_all: list, ids_ignore: list, df_info: pandas.DataFrame, train_pct: float):
    # Get samples that can be included in both datasets
    ids_common = list(set(ids_all).intersection(ids_ignore))  # in case some IMG IDs are not available in main dataset
    ids_available = list(set(ids_all) - set(ids_common))
    df_split = df_info[df_info['case_id'].isin(ids_available)]

    # Get samples that are unique in their category. Only include in train
    class_counts = df_split['GROUP'].value_counts()
    unique_ids = df_split[df_split['GROUP'].isin(class_counts[class_counts == 1].index)]
    df_split = df_split.drop(unique_ids.index)

    # Recalculate train set size, based on samples that must be on it (avoid neg values)
    train_pct_adj = max(0.0, ((train_pct*len(ids_all))-len(ids_common)-len(unique_ids))/len(df_split))

    # Split DF split, based on recalculated ratios and given info
    df_train_part, df_test = train_test_split(df_split, test_size=1-train_pct_adj, stratify=df_split['GROUP'], random_state=25)


    ids_train = df_train_part['case_id'].tolist() + ids_common + unique_ids['case_id'].tolist()
    ids_test = df_test['case_id'].tolist()

    return ids_test, ids_train


def get_sample_from_data_file(inpath: str):
    df = pandas.read_csv(inpath, header=0, index_col=0, usecols=[0, 1], sep=',')
    print(f'* Input DF size: {df.shape}')

    return list(set(df['case_id'].tolist()))


def filter_data_and_save(inpath: str, samplelist: list, outpath: str):
    df = pandas.read_csv(inpath, header=0, index_col=0)
    df = df.loc[df['case_id'].isin(samplelist), :]
    df.reset_index(drop=True, inplace=True)
    df.to_csv(outpath, sep=',', index=True)

    return df


def split_samples_and_save(sample_list: list, n_splits: int, df_info: pandas.DataFrame, out_dir: str):
    # Initialize KFold with N splits (STRATIFIED)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=25)  # random state asegura siempre mismo split

    # Define input DF
    df_split = df_info[df_info['case_id'].isin(sample_list)]

    # Create dir if not exists
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    print('* Data splits:')
    for fold, (train_indices, val_indices) in enumerate(skf.split(df_split, df_split['GROUP'])):
        # Creates train/val groups from DF
        df_train = df_split.iloc[train_indices]
        df_val = df_split.iloc[val_indices]

        # Keep only train/val sample names
        train_set = df_train['case_id'].tolist()
        val_set = df_val['case_id'].tolist()

        print(f'  > FOLD {fold}: Train ({len(train_set)}) & Val ({len(val_set)})')

        # Store train/val data in a single DF
        split_df = pandas.DataFrame({'train': train_set,
                                     'val': val_set + [None]*(len(train_set) - len(val_set))})
        split_df.to_csv(os.path.join(out_dir, f'splits_{fold}.csv'), sep=',', index=True, header=True)


def count_subtype(df, code):
    count = df['oncotree_code'].value_counts().get(code, 0)
    return count


# =========== MAIN
if __name__ == "__main__":
    main_dir = r''  # where raw data is found + where to store new files

    # Define input files
    in_file_img = os.path.join(main_dir, 'IMAGE_sample_ids.txt')
    in_file_data = os.path.join(main_dir, 'tcga_brca_all.csv')
    in_file_clin_aux = os.path.join(main_dir, 'tcga_brca_clin.csv')

    # Define output files
    out_file_img = os.path.join(main_dir, 'DATASET_FINAL', 'IMAGEN_sample_ids.txt')
    out_file_train = os.path.join(main_dir, 'DATASET_FINAL', 'tcga_brca_TRAIN.txt')
    out_file_test = os.path.join(main_dir, 'DATASET_FINAL', 'tcga_brca_TEST.txt')
    out_dir_splits = os.path.join(main_dir, 'DATASET_FINAL', 'data_splits')

    # Run data prep functions
    # # Get subtype and survival data per sample
    df_sample_features = get_feature_df(in_file_data, in_file_clin_aux)

    # # Get sample IDs (in IMG, in all, diff, TEST, TRAIN+VAL)
    sample_ids_img = get_sample_from_img_name(in_file_img, out_file_img)    # Annotated images available
    sample_ids_all = get_sample_from_data_file(in_file_data)                # Data from IMG and all omics available
    sample_ids_not_img = list(set(sample_ids_all) - set(sample_ids_img))    # All data available, excluding annotated IMGs
    sample_ids_test = get_random_set(sample_ids_not_img, len(sample_ids_all), 20)   # 20% of all samples, for final testing
    sample_ids_5fold = list(set(sample_ids_all) - set(sample_ids_test))     # 80% of all samples, for train and validation
    sample_ids_test, sample_ids_5fold = split_train_test(sample_ids_all, sample_ids_img, df_sample_features, 0.8)

    # # Prep filtered data file
    train_df = filter_data_and_save(in_file_data, sample_ids_5fold, out_file_train)
    test_df = filter_data_and_save(in_file_data, sample_ids_test, out_file_test)

    # # Generate data splits
    split_samples_and_save(sample_ids_5fold, 5, df_sample_features, out_dir_splits)
