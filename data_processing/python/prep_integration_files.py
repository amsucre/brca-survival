from pathlib import Path

import pandas


# ===== AUX FUNCTIONS
def merge_predictions_and_save(in_dir: Path, out_f: Path, in_aux: str, get_avg: bool = False):
    # Get available prediction files
    pred_files = [f for f in in_dir.iterdir() if str(f.name).startswith(in_aux)]

    # Open each file, get predictions and append to same DF
    df_merged = pandas.DataFrame()
    for f in pred_files:
        # Read file
        df = pandas.read_csv(f, sep='\t', header=0, index_col=0)

        # Process predictions based on data type
        if '4Q' in in_aux:
            # Split preds + Create one col per prediction + Rename + Convert to float
            df['PREDICTION'] = df['PREDICTION'].str.split(', ')
            df_pred = pandas.DataFrame(df['PREDICTION'].tolist(), index=df.index)
            n_cols = df_pred.shape[1]
            df_pred.columns = [f'Q{i}' for i in range(1, n_cols+1)]
            df_pred = df_pred.astype(float)
        elif 'risk' in in_aux:
            # Just a single column, so copy and change colname
            df_pred = df.copy()
            df_pred.columns = ['RISK']
        elif 'pen_layer' in in_aux:
            # Columns already splitted, so copy df
            df_pred = df.copy()
        elif 'prob' in in_aux:
            # Just a single column, so copy and change colname
            df_pred = df.copy()
            df_pred.columns = ['PROB']
        elif 'class' in in_aux:
            # Just a single column, so copy and change colname
            df_pred = df.copy()
            df_pred.columns = ['CLASS']
        else:
            df_pred = pandas.DataFrame()

        # Merge preds (Append rows or columns)
        if df_merged.empty:
            df_merged = df_pred.copy()
        else:
            df_merged = pandas.concat([df_merged, df_pred])

        print(df_merged.shape)

    # For testing set, get AVG by sample/feature
    if get_avg:
        df_merged = df_merged.groupby(level=0).mean()

    print('FINAL: ', df_merged.shape)

    # Save DF to file
    df_merged.to_csv(out_f, sep='\t', header=True, index=True)


def merge_omis_and_save(in_list: list[tuple[str, Path]], aux_f: Path, out_f: Path,
                        aux_cols: list = ['case_id', 'OUTCOME', 'MONTHS', 'IS_CENSORED'],
                        drop_cols: list = None):
    # Out DF
    merged_df = pandas.DataFrame()

    # Read each file and merge
    for tp in in_list:
        omi = tp[0]  # name
        f = tp[1]    # fpath
        omi_df = pandas.read_csv(f, sep='\t', header=0, index_col=0)

        if drop_cols:  # Lo usamos solo para early INT, para no tener repetidas las "aux_cols"
            omi_df.drop(columns=drop_cols, inplace=True, errors='ignore')

        omi_df.columns = [f'{x}_{omi}' for x in omi_df.columns]

        if merged_df.empty:
            merged_df = omi_df.copy()
        else:
            merged_df = merged_df.merge(omi_df, left_index=True, right_index=True, how='inner')

        print(omi, merged_df.shape)

    # Append extra info: OUTCOME, MONTHS, IS_CENSORED (DEFAULT) ||
    aux_df = pandas.read_csv(aux_f, sep='\t', header=0, index_col=0, usecols=aux_cols)
    merged_df = aux_df.merge(merged_df, left_index=True, right_index=True, how='right')

    # Store file
    merged_df.to_csv(out_f, sep='\t', header=True, index=True)


def merge_files_and_save(in_list: list[Path], out_f: Path):
    merged_df = None
    for fpath in in_list:
        df = pandas.read_csv(fpath, sep='\t', header=0, index_col=['SAMPLE', 'OUTCOME', 'MONTHS', 'IS_CENSORED'])

        if merged_df is None:
            merged_df = df.copy()
        else:
            merged_df = pandas.merge(merged_df, df, left_index=True, right_index=True, how='inner')

        print(merged_df.shape)

    merged_df.to_csv(out_f, sep='\t', index=True, header=True)


# =========== MAIN
if __name__ == "__main__":
    # # Define paths
    result_dir = Path('')
    out_dir = result_dir / 'PRED_MERGED'
    temp_dir = out_dir / 'temp_aux'
    info_file = Path('')  # CLIN DATA FILE
    img_dir = result_dir / 'IMG'
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    pred_list = [('4Q_raw', '4Q_raw_pred')]
    
    # ===== MERGE ALL OMIS + CLIN DATA =====
    test_list = [('SNP', 'final'),
                 ('EXP', 'final'),
                 ('CNV', 'final'),
                 ('MIR', 'final'),
                 ('CLIN', 'final')
                 ]
    
    # For each kind of prediction, create one merged file per omic, and one final merged file
    for p in pred_list:
        name_out = p[0]
        name_in = p[1]
        out_files = list()
        out_files_T = list()
        
        # For each OMI create merged files (V+T)
        for t in test_list:
            omi = t[0]
            test = t[1]
            omi_dir = result_dir / omi / test / 'PREDICTIONS'
    
            print(f'\n===== {omi} =====\n')
    
            # VALIDATION DATA
            out_path = temp_dir / f'{omi}_{name_out}_prediction.tsv'
            merge_predictions_and_save(omi_dir, out_path, 'val_' + name_in)
            out_files.append((omi, out_path))
    
            print('=========')
    
            # TEST DATA
            out_path = temp_dir / f'{omi}_{name_out}_prediction_TEST.tsv'
            merge_predictions_and_save(omi_dir, out_path, 'test_' + name_in, get_avg=True)
            out_files_T.append((omi, out_path))
        
        # Merge all predictions in a single file (x2)
        print('\n===== FINAL MERGE =====\n')
        merge_omis_and_save(out_files, info_file, out_dir / f'OMI_CLIN_{name_out}_prediction.tsv')
        print('=========')
        merge_omis_and_save(out_files_T, info_file, out_dir / f'OMI_CLIN_{name_out}_prediction_TEST.tsv')
    
    # === MERGE ONLY OMIS (AFTER INDIVIDUAL PREP)
    test_list = ['SNP',
                 'EXP',
                 'CNV',
                 'MIR',
                 ]
    
    for p in pred_list:
        name_out = p[0]
        out_files = [(omi, temp_dir / f'{omi}_{name_out}_prediction.tsv') for omi in test_list]
        out_files_T = [(omi, temp_dir / f'{omi}_{name_out}_prediction_TEST.tsv') for omi in test_list]
    
        # Merge all predictions in a single file (x2)
        print('\n===== FINAL MERGE =====\n')
        merge_omis_and_save(out_files, info_file, out_dir / f'OMI_{name_out}_prediction.tsv')
        print('=========')
        merge_omis_and_save(out_files_T, info_file, out_dir / f'OMI_{name_out}_prediction_TEST.tsv')
    
    
    # === MERGE OMI+IMG && OMI+CLIN+IMG (AFTER INDIVIDUAL PREP)
    test_list = ['SNP',
                 'EXP',
                 'CNV',
                 'MIR',
                 'CLIN'
                 # IMG
                 ]
    
    for p in pred_list:
        name_out = p[0]
        out_files = [(omi, temp_dir / f'{omi}_{name_out}_prediction.tsv') for omi in test_list]
        out_files_T = [(omi, temp_dir / f'{omi}_{name_out}_prediction_TEST.tsv') for omi in test_list]
    
        # Merge all predictions in a single file (x2 each)
        # (1) OMI+CLIN+RESNET
        train_file = [('RESNET', img_dir / 'RESNET' / f'IMG_{name_out}_prediction_TRAIN.tsv')]
        test_file = [('RESNET', img_dir / 'RESNET' / f'IMG_{name_out}_prediction_TEST.tsv')]
    
        print(f'\n===== FINAL MERGE: OMI+CLIN+RESNET | {name_out} =====\n')
        merge_omis_and_save(train_file + out_files, info_file, out_dir / f'OMI_CLIN_RESNET_{name_out}_prediction.tsv')
        print('=========')
        merge_omis_and_save(test_file + out_files_T, info_file, out_dir / f'OMI_CLIN_RESNET_{name_out}_prediction_TEST.tsv')
    
        # (2) OMI+RESNET
        train_file = [('RESNET', img_dir / 'RESNET' / f'IMG_{name_out}_prediction_TRAIN.tsv')]
        test_file = [('RESNET', img_dir / 'RESNET' / f'IMG_{name_out}_prediction_TEST.tsv')]
    
        print(f'\n===== FINAL MERGE: OMI+RESNET | {name_out} =====\n')
        merge_omis_and_save(train_file + out_files[:-1], info_file, out_dir / f'OMI_RESNET_{name_out}_prediction.tsv')
        print('=========')
        merge_omis_and_save(test_file + out_files_T[:-1], info_file, out_dir / f'OMI_RESNET_{name_out}_prediction_TEST.tsv')
    
    # === MERGE CLIN+IMG (AFTER INDIVIDUAL PREP)
    for p in pred_list:
        name_out = p[0]
        out_files = [('CLIN', temp_dir / f'CLIN_{name_out}_prediction.tsv')]
        out_files_T = [('CLIN', temp_dir / f'CLIN_{name_out}_prediction_TEST.tsv')]
    
        # Merge all predictions in a single file (x2 each)
        # CLIN+RESNET
        train_file = [('RESNET', img_dir / 'RESNET' / f'IMG_{name_out}_prediction_TRAIN.tsv')]
        test_file = [('RESNET', img_dir / 'RESNET' / f'IMG_{name_out}_prediction_TEST.tsv')]
    
        print(f'\n===== FINAL MERGE: CLIN+RESNET | {name_out} =====\n')
        merge_omis_and_save(train_file + out_files, info_file, out_dir / f'CLIN_RESNET_{name_out}_prediction.tsv')
        print('=========')
        merge_omis_and_save(test_file + out_files_T, info_file, out_dir / f'CLIN_RESNET_{name_out}_prediction_TEST.tsv')
    
    # PREP DATA FOR EARLY INTEGRATION
    # # OMI + CLIN
    raw_dir = Path('')
    
    out_files = [('CLIN', raw_dir / f'BRCA_CLIN_train.csv'),
                 ('SNP', raw_dir / f'BRCA_SNP_train.csv'),
                 ('EXP', raw_dir / f'BRCA_RNA_train.csv'),
                 ('CNV', raw_dir / f'BRCA_CNV_train.csv'),
                 ('MIR', raw_dir / f'BRCA_MIR_train.csv'),
                 ]
    out_files_T = [(omi, raw_dir / file.name.replace('train', 'test')) for (omi, file) in out_files]
    
    drop = ['OUTCOME', 'MONTHS', 'IS_CENSORED']
    
    # Merge all predictions in a single file (x2)
    print('\n===== FINAL MERGE (O+C) =====\n')
    merge_omis_and_save(out_files, info_file, out_dir / f'OMI_CLIN_EARLY.tsv', drop_cols=drop)
    print('=========')
    merge_omis_and_save(out_files_T, info_file, out_dir / f'OMI_CLIN_EARLY_TEST.tsv', drop_cols=drop)
    
    print('\n===== FINAL MERGE (OMI) =====\n')
    merge_omis_and_save(out_files[1:], info_file, out_dir / f'OMI_EARLY.tsv', drop_cols=drop)
    print('=========')
    merge_omis_and_save(out_files_T[1:], info_file, out_dir / f'OMI_EARLY_TEST.tsv', drop_cols=drop)
    
    # # OMI + CLIN + IMG
    raw_dir = Path('')
    
    out_files = [('CLIN', raw_dir / f'BRCA_CLIN_train.csv'),
                 ('SNP', raw_dir / f'BRCA_SNP_train.csv'),
                 ('EXP', raw_dir / f'BRCA_RNA_train.csv'),
                 ('CNV', raw_dir / f'BRCA_CNV_train.csv'),
                 ('MIR', raw_dir / f'BRCA_MIR_train.csv'),
                 ]
    out_files_T = [(omi, raw_dir / file.name.replace('train', 'test')) for (omi, file) in out_files]
    
    out_files_img = [('EARLY', img_dir / 'EARLY' / f'IMG_early_prediction_TRAIN.tsv')]
    out_files_img_T = [('EARLY', img_dir / 'EARLY' / f'IMG_early_prediction_TEST.tsv')]
    
    drop = ['OUTCOME', 'MONTHS', 'IS_CENSORED']
    
    # Merge all predictions in a single file (x2)
    for train, test in zip(out_files_img, out_files_img_T):
        print(f'\n===== FINAL MERGE (O+C+I) [{train[0]}]=====\n')
        merge_omis_and_save(out_files+[train], info_file, out_dir / f'OMI_CLIN_{train[0]}_EARLY.tsv', drop_cols=drop)
        print('=========')
        merge_omis_and_save(out_files_T+[test], info_file, out_dir / f'OMI_CLIN_{test[0]}_EARLY_TEST.tsv', drop_cols=drop)
