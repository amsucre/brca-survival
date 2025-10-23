import ast
import re
from pathlib import Path

import pandas
import torch


# ================= AUX =================
def data_to_file(fpath: Path, data_list: list[float], sa_id: str, sl_id: str, is_new: bool):
    str_to_write = sa_id + '\t' + sl_id + '\t' + '\t'.join([str(x) for x in data_list]) + '\n'

    if is_new:
        first_row = '\t'.join(['SAMPLE', 'SLIDE'] + [str(x) for x in range(len(data_list))]) + '\n'
        with open(fpath, 'w') as f:
            f.write(first_row)
            f.write(str_to_write)
        is_new = False
    else:
        with open(fpath, 'a') as f:
            f.write(str_to_write)

    return is_new


def merge_samples(in_path: Path, out_path: Path):
    df = pandas.read_csv(in_path, sep='\t', header=0)
    if 'SLIDE' in df.columns:
        df.drop(columns=['SLIDE'], inplace=True)
    df_group = df.groupby(by='SAMPLE', as_index=True).mean()

    # Save new DF
    df_group.to_csv(out_path, sep='\t', index=True, header=True)


def get_pred_data(fpath: Path):
    df = pandas.read_csv(fpath, sep=',', header=0, index_col=0, usecols=['slide_id', 'model_output', 'risk'])
    df.index.name = 'SAMPLE'
    df['model_output'] = df['model_output'].apply(ast.literal_eval)

    # Get risk (keep as given)
    serie_risk = df['risk']
    serie_risk.name = 'RISK'

    # Get 4q-raw + Split list in cols
    df_4q_raw = pandas.DataFrame(df['model_output'].tolist(), index=df.index, columns=['Q1R', 'Q2R', 'Q3R', 'Q4R'])

    # Get 4q (using softmax) + Split list in cols
    tensor = torch.tensor(df['model_output'].tolist())
    prob_tensor = torch.softmax(tensor, dim=1)
    df_4q = pandas.DataFrame(prob_tensor.tolist(), index=df.index, columns=['Q1', 'Q2', 'Q3', 'Q4'])

    return df_4q, df_4q_raw, serie_risk


def get_pct_from_file_list(pct_file_list: list[Path], split_file_list: list[Path], test_file: Path):
    # Get test samples
    test_samples = pandas.read_csv(test_file, sep=',', header=0, usecols=['case_id'])['case_id'].tolist()
    test_samples = list(set(test_samples))
    print('* Test samples (n):', len(test_samples), test_samples)

    col_keep = ['case_id', 'tumor', 'stroma', 'other']
    merged_df = None
    merged_test_df = None
    for p, s in zip(pct_file_list, split_file_list):
        # Read fold data
        p_df = pandas.read_csv(p, sep=',', header=0, usecols=col_keep, index_col='case_id')
        s_df = pandas.read_csv(s, sep=',', header=0, usecols=['val'])

        # Get samples to keep (from split file)
        keep_samples = [x for x in set(s_df['val'].tolist()) if not pandas.isna(x)]  # Set drops NAs
        print('\n* Samples to keep (n): ', len(keep_samples))

        # Get all samples (njust informative)
        all_samples = p_df.index.tolist()
        print('* All samples (n): ', len(all_samples), len(list(set(all_samples))))

        # Filter dataset
        keep_df = p_df.loc[keep_samples, :]

        kept_samples = keep_df.index.tolist()
        print('* Kept samples (n): ', len(kept_samples), len(list(set(kept_samples))))

        # Drop duplicate samples
        keep_df = keep_df[~keep_df.index.duplicated(keep='first')]

        print('* Final size: ', keep_df.shape)

        # Concat all DFs
        if merged_df is None:
            merged_df = keep_df.copy()
        else:
            merged_df = pandas.concat([merged_df, keep_df])

        # Filter dataset (TEST)
        test_df = p_df.loc[test_samples, :]

        test_samples = test_df.index.tolist()
        print('* Test samples (n): ', len(test_samples), len(list(set(test_samples))))

        # Drop duplicate samples (TEST)
        test_df = test_df[~test_df.index.duplicated(keep='first')]

        print('* Test Final size: ', test_df.shape)

        # Concat all DFs (TEST)
        if merged_test_df is None:
            merged_test_df = test_df.copy()
        else:
            merged_test_df = pandas.concat([merged_test_df, test_df])

    # Merge TEST data (5 copies per each)
    final_test_df = merged_test_df.groupby(merged_test_df.index).mean()

    return merged_df, final_test_df


# =========== MAIN
if __name__ == "__main__":
    # Define input dirs / output dirs
    img_home = Path('')

    uni_dir = img_home / 'RESULTS_TRAINING_UNI' / ''
    resnet_dir = img_home / 'RESULTS_TRAINING_RESNET' / ''

    split_dir = img_home / 'data_splits' / 'splits_segmentation'

    out_dir = Path(img_home / 'DATA_INT')
    out_uni_dir = out_dir / 'UNI'
    out_resnet_dir = out_dir / 'RESNET'

    out_uni_dir.mkdir(parents=True, exist_ok=True)
    out_resnet_dir.mkdir(parents=True, exist_ok=True)

    # ===== EXTRACT VALUES FOR "4Q-RAW" MODELS | FOUNDATIONAL (UNI) =====
    print('\n========= UNI files =========')
    
    # # List VAL & TEST files
    fund_val_files = [x for x in uni_dir.iterdir() if x.is_file() and x.name.startswith('val_results_')]
    fund_test_files = [x for x in uni_dir.iterdir() if x.is_file() and x.name.startswith('test_results_')]
    
    # # Define output files
    out_dir_t = out_uni_dir / 'temp'  # only for test
    out_dir_t.mkdir(parents=True, exist_ok=True)
    
    val_4q_r_f = out_uni_dir / 'IMG_4Q_raw_prediction_TRAIN.tsv'
    test_4q_r_f = out_dir_t / 'IMG_4Q_raw_prediction_TEST_all.tsv'
    
    # # Iter files and extract data
    new = True
    for f in fund_val_files:
        _, val_4q_r, _ = get_pred_data(f)
    
        if new:
            val_4q_r.to_csv(val_4q_r_f, sep='\t', index=True, header=True)
            new = False
        else:
            val_4q_r.to_csv(val_4q_r_f, sep='\t', index=True, header=False, mode='a')
        
    new = True
    for f in fund_test_files:
        _, test_4q_r, _ = get_pred_data(f)
    
        if new:
            test_4q_r.to_csv(test_4q_r_f, sep='\t', index=True, header=True)
            new = False
        else:
            test_4q_r.to_csv(test_4q_r_f, sep='\t', index=True, header=False, mode='a')
    
    # # Merge TEST files
    print('\n========= FOUNDATIONAL files (Merge) =========')
    files_to_merge = [test_4q_r_f]
    for in_f in files_to_merge:
        # Define output file
        out_f = out_uni_dir / in_f.name.replace('_all', '')
    
        # Read input + Merge rows from the same sample
        merge_samples(in_f, out_f)

    # ===== EXTRACT VALUES FOR "4Q-RAW" MODELS | FOUNDATIONAL (RESNET) =====
    print('\n========= RESNET files =========')
    
    # # List VAL & TEST files
    resnet_val_files = [x for x in resnet_dir.iterdir() if x.is_file() and x.name.startswith('val_results_')]
    resnet_test_files = [x for x in resnet_dir.iterdir() if x.is_file() and x.name.startswith('test_results_')]
    
    # # Define output files
    out_dir_t = out_resnet_dir / 'temp'  # only for test
    out_dir_t.mkdir(parents=True, exist_ok=True)
    
    val_4q_r_f = out_resnet_dir / 'IMG_4Q_raw_prediction_TRAIN.tsv'
    test_4q_r_f = out_dir_t / 'IMG_4Q_raw_prediction_TEST_all.tsv'
    
    # # Iter files and extract data
    new = True
    for f in resnet_val_files:
        _, val_4q_r, _ = get_pred_data(f)
    
        if new:
            val_4q_r.to_csv(val_4q_r_f, sep='\t', index=True, header=True)
            new = False
        else:
            val_4q_r.to_csv(val_4q_r_f, sep='\t', index=True, header=False, mode='a')
        
    new = True
    for f in resnet_test_files:
        _, test_4q_r, _ = get_pred_data(f)
    
        if new:
            test_4q_r.to_csv(test_4q_r_f, sep='\t', index=True, header=True)
            new = False
        else:
            test_4q_r.to_csv(test_4q_r_f, sep='\t', index=True, header=False, mode='a')
    
    # # Merge TEST files
    print('\n========= RESNET files (Merge) =========')
    files_to_merge = [test_4q_r_f]
    for in_f in files_to_merge:
        # Define output file
        out_f = out_resnet_dir / in_f.name.replace('_all', '')
    
        # Read input + Merge rows from the same sample
        merge_samples(in_f, out_f)


    # ===== PREP PCTS FOR EARLY INTEGRATION =====
    pct_dir = Path('')  # file were predicted tissue-types % per img are stored
    pct_files = [pct_dir / f'tcga_predicted_img_pct_{i}.csv' for i in range(1, 6)]
    
    split_dir = Path('')
    split_files = [split_dir / f'splits_{i}.csv' for i in range(5)]
    
    data_dir = Path('')
    test_file = data_dir / 'tcga_brca_TEST.txt'
    
    train_df, test_df = get_pct_from_file_list(pct_files, split_files, test_file)
    
    train_path = out_dir / 'EARLY' / 'IMG_early_prediction_TRAIN.tsv'
    train_df.to_csv(train_path, sep='\t', header=True, index=True)
    
    test_path = out_dir / 'EARLY' / 'IMG_early_prediction_TEST.tsv'
    test_df.to_csv(test_path, sep='\t', header=True, index=True)
