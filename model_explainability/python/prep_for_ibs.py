from pathlib import Path

import pandas
import torch


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


def split_pred_file(in_file: Path, out_dir: Path, keys: list, which: str):
    df = pandas.read_csv(in_file, sep='\t', header=0, index_col=0)
    
    # Get + save real data
    df_real = df.loc[:, ['OUTCOME', 'MONTHS', 'IS_CENSORED']]
    df_real.to_csv(out_dir / f'real_survival_{which}.tsv', sep='\t', index=True, header=True)
    
    # Get + save predictions (per Q and cumulative)
    for k in keys:
      print('\n\n=====', k, '=====')
      df_k = df.loc[:, [f'Q1_{k}', f'Q2_{k}', f'Q3_{k}', f'Q4_{k}']]
      df_k.columns = ['Q1', 'Q2', 'Q3', 'Q4']  # sea siempre igual
      df_k.to_csv(out_dir / f'pred_death_{k}_{which}.tsv', sep='\t', index=True, header=True)
      
      df_surv = df_k.map(lambda x: 1-x)
      df_surv.to_csv(out_dir / f'pred_surv_{k}_{which}.tsv', sep='\t', index=True, header=True)

      df_surv_cum = df_k.cumsum(axis=1)
      df_surv_cum = df_surv_cum.map(lambda x: 1-x)
      df_surv_cum['Q4'] = 0
      df_surv_cum.to_csv(out_dir / f'pred_surv_cum_{k}_{which}.tsv', sep='\t', index=True, header=True)
      

def extract_pred_ind_omi_files(in_dir: Path, out_dir: Path, keys: list, which_set: str, which_stat: str):
    for k in keys:
        p = in_dir / k[0] / k[1] / 'PREDICTIONS' if k[1] else in_dir / k[0] / 'PREDICTIONS'
        od = out_dir / k[0]
        od.mkdir(parents=True, exist_ok=True)
        
        for i in range(5):
            # Get predictions
            in_f = p / f'{which_set.lower()}_{which_stat}_pred_{i}.tsv'
            df = pandas.read_csv(in_f, sep='\t', header=0, index_col=0)
        
            df['PREDICTION'] = df['PREDICTION'].str.split(', ')
            df = pandas.DataFrame(df['PREDICTION'].tolist(), index=df.index)
            n_cols = df.shape[1]
            df.columns = [f'Q{i}' for i in range(1, n_cols+1)]
            df = df.astype(float)
            
            # Save raw preds
            df.to_csv(od / f'pred_death_{k[0]}_{i}_{which_set}.tsv', sep='\t', index=True, header=True)

            # Get stats
            df_surv = df.map(lambda x: 1-x)
            df_surv.to_csv(od / f'pred_surv_{k[0]}_{i}_{which_set}.tsv', sep='\t', index=True, header=True)

            df_surv_cum = df.cumsum(axis=1)
            df_surv_cum = df_surv_cum.map(lambda x: 1-x)
            df_surv_cum['Q4'] = 0
            df_surv_cum.to_csv(od / f'pred_surv_cum_{k[0]}_{i}_{which_set}.tsv', sep='\t', index=True, header=True)


def extract_pred_ind_img_files(in_dir: Path, out_dir: Path, which_set: str, which_stat: str, which_input: str = 'IMG'):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(5):
        # Get predictions
        in_f = in_dir / f'{which_set.lower()}_results_fold_{i}.csv'
        if in_f.exists():
            print(f'* {in_f} exists')     
            df_4q, df_4q_raw, _ = get_pred_data(in_f)
            
            if which_stat == '4Q':
                df = df_4q.copy()
            else:
                df = df_4q_raw.copy()
                        
            # Save raw preds
            df.to_csv(out_dir / f'pred_death_{which_input}_{i}_{which_set}.tsv', sep='\t', index=True, header=True)

            # Get stats
            df_surv = df.map(lambda x: 1-x)
            df_surv.to_csv(out_dir / f'pred_surv_{which_input}_{i}_{which_set}.tsv', sep='\t', index=True, header=True)

            df_surv_cum = df.cumsum(axis=1)
            df_surv_cum = df_surv_cum.map(lambda x: 1-x)
            df_surv_cum['Q4'] = 0
            df_surv_cum.to_csv(out_dir / f'pred_surv_cum_{which_input}_{i}_{which_set}.tsv', sep='\t', index=True, header=True)
        else:
            print(f'File {in_f} does not exist')


def extract_pred_int_files(in_dir: Path, out_dir: Path, keys: list, which_set: str, which_stat: str):
    for k in keys:
        print(k)
        p = in_dir / k[1] / 'PREDICTIONS'
        
        od_each = out_dir / 'INT_EACH' / k[0]
        od_each.mkdir(parents=True, exist_ok=True)
        
        od_mean = out_dir / 'INT_MEAN'
        od_mean.mkdir(parents=True, exist_ok=True)
                
        all_preds = list()
        
        for i in range(5):
            # Get predictions
            in_f = p / f'{which_set.lower()}_{which_stat}_pred_{i}.tsv'
            df = pandas.read_csv(in_f, sep='\t', header=0, index_col=0)
        
            df['PREDICTION'] = df['PREDICTION'].str.split(', ')
            df = pandas.DataFrame(df['PREDICTION'].tolist(), index=df.index)
            n_cols = df.shape[1]
            df.columns = [f'Q{i}' for i in range(1, n_cols+1)]
            df = df.astype(float)
            
            # Save raw preds
            df.to_csv(od_each / f'pred_death_{k[0]}_{i}_{which_set}.tsv', sep='\t', index=True, header=True)

            # Get stats
            df_surv = df.map(lambda x: 1-x)
            df_surv.to_csv(od_each / f'pred_surv_{k[0]}_{i}_{which_set}.tsv', sep='\t', index=True, header=True)

            df_surv_cum = df.cumsum(axis=1)
            df_surv_cum = df_surv_cum.map(lambda x: 1-x)
            df_surv_cum['Q4'] = 0
            df_surv_cum.to_csv(od_each / f'pred_surv_cum_{k[0]}_{i}_{which_set}.tsv', sep='\t', index=True, header=True)
            
            # Save preds to merge
            all_preds.append(df)
        
        # Merge preds and get AVG
        merged_pred_df = pandas.concat(all_preds, ignore_index=False)
        mean_pred_df = merged_pred_df.groupby(merged_pred_df.index).mean()
        
        # Save raw preds
        mean_pred_df.to_csv(od_mean / f'pred_death_{k[0]}_{which_set}.tsv', sep='\t', index=True, header=True)

        # Get stats
        df_surv = mean_pred_df.map(lambda x: 1-x)
        df_surv.to_csv(od_mean / f'pred_surv_{k[0]}_{which_set}.tsv', sep='\t', index=True, header=True)

        df_surv_cum = mean_pred_df.cumsum(axis=1)
        df_surv_cum = df_surv_cum.map(lambda x: 1-x)
        df_surv_cum['Q4'] = 0
        df_surv_cum.to_csv(od_mean / f'pred_surv_cum_{k[0]}_{which_set}.tsv', sep='\t', index=True, header=True)
            
          
# =========== MAIN
if __name__ == "__main__":
    # Define paths
    main_dir = Path('')   # Results DIR
    
    # Get individual prediction files from 5-fold files (Each prediction - OMIs)
    omi_list = [('SNP', ''),
                 ('EXP', ''),
                 ('CNV', ''),
                 ('MIR', ''),
                 ('CLIN', ''), 
                ]
    out_dir = main_dir / 'IBS' / 'IN' / 'IND_EACH'
    extract_pred_ind_omi_files(main_dir, out_dir, omi_list, 'TEST', '4Q')
    
    # Get individual prediction files from 5-fold files (Each prediction - IMG)
    img_dir = Path('')  # Results DIR (IMG models)
    out_dir = main_dir / 'IBS' / 'IN' / 'IND_EACH' / 'IMG'
    extract_pred_ind_img_files(img_dir, out_dir, 'TEST', '4Q')
    
    # Get integration prediction files from 5-fold files (Each + Mean predictions)
    in_dir = main_dir / 'INTEGRATION'
    out_dir = main_dir / 'IBS' / 'IN'
    models = [('OM', 'OMI_4Q_raw'),
              ('OM_CL', 'OMI_CLIN_4Q_raw'), 
              ('OM_IM', 'OMI_RESNET_4Q_raw'), 
              ('CL_IM', 'CLIN_RESNET_4Q_raw'), 
              ('OM_CL_IM', 'OMI_CLIN_RESNET_4Q_raw'), 
              ('OM_E', 'OMI_EARLY'), 
              ('OM_CL_E', 'OMI_CLIN_EARLY'), 
              ('OM_CL_IM_E', 'OMI_CLIN_RESNET_MEAN_EARLY')]
    extract_pred_int_files(in_dir, out_dir, models, 'TEST', '4Q')
    
    # Get individual prediction files from 5-fold files (Each prediction - SOA)
    soa_dir = Path('')  # Results DIR (SOA models)
    out_dir = main_dir / 'IBS' / 'IN' / 'SOA_EACH'
    soa_list = [('PORPOISE', soa_dir / ''),   # Path to specific PORPOISE results dir
                ('MGCT', soa_dir / ''),       # Path to specific MGCT results dir
                ('MCAT', soa_dir / '')]       # Path to specific MCAT results dir
    for t in soa_list:
        extract_pred_ind_img_files(t[1], out_dir / t[0], 'TEST', '4Q', t[0])
    