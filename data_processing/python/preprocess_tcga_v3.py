import os
import requests
from pathlib import Path

import pandas


# ===== AUX FUNCTIONS
def process_tcga_data(fpath, key, feature_list=None, sample_list=None, append_aux=True, data_files=None, synonym_dict=None):
    print(f'----- {key} DATA -----')

    # Read file
    df, rows, fields = read_df(fpath, compression='gzip')
    print('* Init Size >', df.shape)

    # Keep relevant fields
    fields_to_keep = dict()
    fields_to_keep['CLIN'] = ['submitter_id', 'gender.demographic', 'vital_status.demographic', 'age_at_index.demographic',
                              'code.tissue_source_site', 'days_to_last_follow_up.diagnoses', 'days_to_death.demographic',
                              'race.demographic', 'ajcc_pathologic_stage.diagnoses', 'age_at_earliest_diagnosis_in_years.diagnoses.xena_derived']
    fields_to_keep['SNP'] = ['sample', 'gene', 'chrom', 'start', 'end', 'ref', 'alt', 'effect', 'dna_vaf']
    fields_to_keep['CNV'] = fields  # columns > samples
    fields_to_keep['RNA'] = fields  # columns > samples
    fields_to_keep['MIR'] = fields  # columns > samples
    fields_to_keep['PROT'] = fields  # columns > samples

    df = df.loc[:, fields_to_keep[key]]

    # Transform data
    if key == 'CLIN':
        df = transform_clinical(df, sample_list, append_aux, data_files)  # DONE
    elif key == 'SNP':
        df = transform_snp(df, feature_list, sample_list)  # DONE
    elif key == 'CNV':
        df = transform_cnv(df, feature_list, sample_list, synonym_dict)  # DONE
    elif key == 'RNA':
        df = transform_exp(df, feature_list, sample_list, synonym_dict)  # DONE
    elif key == 'MIR':
        df = transform_mir(df, feature_list, sample_list)  # DONE
    
    # Sort samples a-z
    df.sort_values('case_id', inplace=True)

    if df.shape[1] < 20:
        pandas.options.display.max_columns = None
    else:
        pandas.options.display.max_columns = 5

    return df, df.case_id.tolist()


def read_df(f, index_col: int | bool = False, xlsx: bool = False, compression: str = None, cols: list = None):
    if xlsx:
        df = pandas.read_excel(f, header=1)
    elif cols:  # specific cols to read
        df = pandas.read_csv(f, sep='\t', index_col=index_col, compression=compression, usecols=cols)
    else:
        df = pandas.read_csv(f, sep='\t', index_col=index_col, compression=compression)
    rows = df.index.tolist()
    cols = df.columns.tolist()
    return df, rows, cols


def transform_clinical(df, sample_list=None, append_aux=True, data_files=None):
    df = df.copy()
    # Transform data: is_female, survival_months,censorship, stage, age-diagnosis
    df['is_female'] = df['gender.demographic'].apply(lambda x: 1 if x == 'female' else 0)

    d_per_m = 365.0 / 12.0
    df['survival_days'] = df.apply(get_survival_days, axis=1)
    df = df.dropna(subset=['survival_days'])  # If not survival data is available, DROP that rows
    df['survival_months'] = df['survival_days'].apply(lambda x: round(x / d_per_m, 2))

    df['censorship'] = df['vital_status.demographic'].apply(lambda x: 0 if x == 'Dead' else 1)

    df['stage'] = df['ajcc_pathologic_stage.diagnoses'].apply(lambda x: x.lstrip('Stage ') if isinstance(x, str) else x)

    df['age_diagnosis'] = df['age_at_earliest_diagnosis_in_years.diagnoses.xena_derived'].apply(lambda x: round(x, 2))

    # NOT AVAILABLE: oncotree_code, PAM50 subtype >> MUST CONSIDER OTHER SOURCES
    if append_aux:
        # Read cBioPortal data to get oncotree_code + drop rows with "BREAST" generic classification
        cols_to_keep = ['Patient ID', 'Oncotree Code']
        df_otree, x, x = read_df(data_files['CLIN_OT'], cols=cols_to_keep)
        df_otree = df_otree.loc[df_otree['Oncotree Code'] != 'BREAST']

        # Merge both DFs
        df = df.merge(df_otree, left_on='submitter_id', right_on='Patient ID', how='left')

        # Read other Xena file to get PAM50 subtype
        cols_to_keep = ['sampleID', 'PAM50Call_RNAseq', 'PAM50_mRNA_nature2012']
        df_pam, x, x = read_df(data_files['CLIN_PAM'], cols=cols_to_keep)
        df_pam = prep_pam50(df_pam)

        # Merge both DFs
        df = df.merge(df_pam, left_on='submitter_id', right_on='case_id', how='left')

        # Keep final columns
        cols_to_keep = ['submitter_id', 'code.tissue_source_site', 'is_female', 'age_at_index.demographic', 'age_diagnosis',
                        'race.demographic', 'stage', 'Oncotree Code', 'pam50', 'survival_months', 'censorship']
        df = df.loc[:, cols_to_keep]
        df.columns = ['case_id', 'site', 'is_female', 'age_index', 'age_diagnosis', 'race', 'stage', 'oncotree_code',
                      'pam50_subtype', 'survival_months', 'censorship']
    else:
        # Keep final columns
        cols_to_keep = ['submitter_id', 'code.tissue_source_site', 'is_female', 'age_at_index.demographic',
                        'age_diagnosis', 'race.demographic', 'stage', 'survival_months', 'censorship']
        df = df.loc[:, cols_to_keep]
        df.columns = ['case_id', 'site', 'is_female', 'age_index',
                      'age_diagnosis', 'race', 'stage', 'survival_months', 'censorship']

    # Filter samples
    if sample_list:
        df = df.loc[df['case_id'].isin(sample_list)]
        print('* After drop samples >', df.shape)

    # Drop duplicated rows
    df.drop_duplicates(inplace=True)  # ok because case id is column

    return df


def transform_snp(df, feature_list=None, sample_list=None):
    # Define case ID
    df['case_id'] = df['sample'].apply(get_case_from_sample)
    df.drop('sample', axis=1, inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    print('* Size after drop duplicates >', df.shape)

    # Get matrix sample vs. mutated gene
    df_pivot = pandas.crosstab(df['case_id'], df['gene'])

    print('* Pivot Init Size >', df_pivot.shape)

    # Filter samples
    if sample_list:
        df_pivot = df_pivot.loc[sample_list, :]
        print('* After drop samples >', df.shape)

    # IF A FEATURE LIST IS GIVEN, FILTER. IF NOT, KEEP ALL
    if feature_list:
        print('* Feature list available...')
        common = list(set(feature_list).intersection(df_pivot.columns.tolist()))
        df_pivot = df_pivot.loc[:, common]

    df_pivot.reset_index(inplace=True, drop=False)

    return df_pivot


def transform_cnv(df, feature_list=None, sample_list=None, synonym_dict=None):
    # Rename index
    df.set_index('Ensembl_ID', inplace=True)

    # Transpose DF
    df = df.T

    # Define case ID
    df['case_id'] = df.index.to_series().apply(get_case_from_sample)
    df.set_index('case_id', inplace=True)

    # Modify gene ID
    df.columns = df.columns.str.split('.').str[0]

    inspect_df(df)

    # Check if cases are duplicated
    index_dup = df.index.duplicated()
    index_dup_n = index_dup.sum()
    print('* Duplicated case IDs (n) >', index_dup_n)

    # Check if genes are duplicated
    col_dup = df.columns.duplicated()
    col_dup_n = col_dup.sum()
    print('* Duplicated genes (n) >', col_dup_n)

    # Drop duplicates if any
    if col_dup_n:
        df_t = df.T.groupby(df.columns).max()  # Prefiere hacer en T que indicar axis=1
        df = df_t.T

    if index_dup_n:
        df = df.groupby(df.index).max()

    print('* Size after drop duplicates >', df.shape)

    # Filter samples
    if sample_list:
        df = df.loc[sample_list, :]
        print('* After drop samples >', df.shape)

    # IF A FEATURE LIST IS GIVEN, FILTER. IF NOT, KEEP ALL
    if feature_list:
        print('* Feature list available...')
        # Get ensembl IDs corresponding to chosen features
        feature_ensembl_list = [get_id_from_ensembl(x, synonym_dict) for x in feature_list]

        # Filter genes to keep only chosen features
        common = list(set(feature_ensembl_list).intersection(df.columns.tolist()))
        df = df.loc[:, common]

    df = df.map(rescale_cnv)

    inspect_df(df)

    print('* Filtered >', df.shape)

    # Sort data
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)

    # Case ID to columns
    df.reset_index(inplace=True, drop=False)

    return df


def transform_exp(df, feature_list=None, sample_list=None, synonym_dict=None):
    # Rename index
    df.set_index('Ensembl_ID', inplace=True)

    # Transpose DF
    df = df.T

    # Define case ID
    df['case_id'] = df.index.to_series().apply(get_case_from_sample)
    df = df.reset_index()  # dejar sample en columnas
    df.set_index('case_id', inplace=True)

    # Drop rows that correspond to NORMAL samples + Drop sample col
    df = df[~df['index'].str.endswith(('11A', '11B'))]
    df.drop('index', axis=1, inplace=True)

    # Modify gene ID
    df.columns = df.columns.str.split('.').str[0]

    # Check if cases are duplicated
    index_dup = df.index.duplicated()
    index_dup_n = index_dup.sum()
    print('* Duplicated case IDs (n) >', index_dup_n)

    # Check if genes are duplicated
    col_dup = df.columns.duplicated()
    col_dup_n = col_dup.sum()
    print('* Duplicated genes (n) >', col_dup_n)

    # Drop duplicates if any
    if col_dup_n:
        df_t = df.T.groupby(df.columns).max()  # Prefiere hacer en T que indicar axis=1
        df = df_t.T

    if index_dup_n:  # Para algunos tenemos 2 muestras. Nos quedamos con la media
        df = df.groupby(df.index).mean()

    print('* Size after drop duplicates >', df.shape)

    # IF A FEATURE LIST IS GIVEN, FILTER. IF NOT, KEEP ALL
    if feature_list:
        print('* Feature list available...')
        # Get ensembl IDs corresponding to chosen features
        feature_ensembl_list = [get_id_from_ensembl(x, synonym_dict) for x in feature_list]

        # Filter genes to keep only chosen features
        common = list(set(feature_ensembl_list).intersection(df.columns.tolist()))
        df = df.loc[:, common]

        # Rename columns
        df.columns = df.columns.to_series().apply(get_name_from_ensembl)

    # Filter samples
    if sample_list:
        df = df.loc[sample_list, :]
        print('* After drop samples >', df.shape)

    # Sort data
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)

    # Case ID to columns
    df.reset_index(inplace=True, drop=False)

    return df


def transform_mir(df, feature_list=None, sample_list=None):
    # Rename index
    df.set_index(df.columns[0], inplace=True)

    # Delete miRNAs where everything is NaN
    df = df.dropna(how='all')
    print('* After miRNA drop >', df.shape)

    # Transpose DF
    df = df.T

    # Define case ID
    df['case_id'] = df.index.to_series().apply(get_case_from_sample)
    df = df.reset_index() 
    df.set_index('case_id', inplace=True)

    # Drop rows that correspond to NORMAL samples + Drop sample col
    df = df[~df['index'].str.endswith(('11A', '11B'))]
    df.drop('index', axis=1, inplace=True)
    print('* After sample (normal) drop >', df.shape)

    # Delete samples where everything is NaN
    df = df.dropna(how='all')
    print('* After sample (empty) drop >', df.shape)

    # Check if cases are duplicated
    index_dup = df.index.duplicated()
    index_dup_n = index_dup.sum()
    print('* Duplicated case IDs (n) >', index_dup_n)

    # Check if miRs are duplicated
    col_dup = df.columns.duplicated()
    col_dup_n = col_dup.sum()
    print('* Duplicated miRNAs (n) >', col_dup_n)

    # Drop duplicates if any
    if col_dup_n:
        df_t = df.T.groupby(df.columns).max()
        df = df_t.T

    if index_dup_n:
        df = df.groupby(df.index).mean()

    print('* Size after drop duplicates >', df.shape)

    # IF A FEATURE LIST IS GIVEN, FILTER. IF NOT, KEEP ALL
    if feature_list:
        print('* Feature list available...')
        # Filter genes to keep only chosen features
        common = list(set(feature_list).intersection(df.columns.tolist()))
        df = df.loc[:, common]

    # Filter samples
    if sample_list:
        df = df.loc[sample_list, :]
        print('* After drop samples >', df.shape)

    # Sort data
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)

    # Case ID to columns
    df.reset_index(inplace=True, drop=False)

    return df



def filter_tcga_data(fpath, key, data_files, synonym_dict=None):
    print(f'----- {key} DATA -----')

    # Read file
    df, _, _ = read_df(fpath, index_col=0)
    print('* Init Size >', df.shape)

    # Filter data
    if key == 'SNP' or key == 'CNV':
        df = filter_df(df, 100, data_files=data_files)
    elif key == 'RNA':
        df = filter_df(df, 0, key, data_files=data_files, synonym_dict=synonym_dict)

    # Rename genes
    if key == 'CNV' or key == 'RNA':
        df.columns = df.columns.to_series().apply(get_name_from_ensembl)

    # Print results
    if df.shape[1] < 50:
        pandas.options.display.max_columns = None
    else:
        pandas.options.display.max_columns = 5

    print('* Final Size >', df.shape)
    print('* Head ')
    print(df.head(5))

    return df


def get_survival_days(row):
    if row['vital_status.demographic'] == 'Alive':
        return row['days_to_last_follow_up.diagnoses']
    else:
        return row['days_to_death.demographic']


def get_case_from_sample(sample_id):
    parts = sample_id.split('-')
    return '-'.join(parts[0:-1])


def filter_cols_by_thr(df, thr, key):
    if key == 'CNV':
        df_count = (df.notna()) & (df != 0)
    else:
        df_count = df.copy()
    samples_per_item = df_count.sum(axis=0)
    items_many_rows = samples_per_item[samples_per_item >= thr].index.tolist()
    items_few_rows = samples_per_item[samples_per_item < thr].index.tolist()

    df = df.loc[:, items_many_rows]

    print(f'* {key} (keep) >', len(items_many_rows))
    print(f'* {key} (drop) >', len(items_few_rows))

    return df


def inspect_df(df):
    print('* Size >', df.shape)

    min_value = df.min().min()
    max_value = df.max().max()

    print('* Min >', min_value)
    print('* Min >', max_value)

    counter = df.stack().value_counts()
    counter.sort_index(inplace=True)
    count_nan = df.isna().sum().sum()
    count_num = counter.sum()

    pandas.options.display.max_rows = None
    print('* Counts')
    print(counter)
    print('Nan >\t', count_nan)
    print('Value >\t', count_num)


def rescale_cnv(x):
    if pandas.isna(x):
        return 0  # NAN > Normal
    elif x == 0:
        return -2  # Complete deletion
    elif x == 1:
        return -1  # Partial deletion
    elif x == 2:
        return 0   # Normal
    elif x == 3:
        return 1   # Gain
    elif x >= 4:
        return 2   # Large gain
    else:
        return None


def get_info_ensembl(rs_id):
    # Define ensembl URL
    url = f'https://rest.ensembl.org/variation/human/{rs_id}'

    # Make request
    response = requests.get(url, headers={'Content-Type': 'application/json'})

    # Check response
    if response.status_code == 200:
        data = response.json()
        # Get data
        chrom = data['mappings'][0]['seq_region_name']
        position = data['mappings'][0]['start']
        allele = data['mappings'][0]['allele_string']
        return chrom, position, allele
    else:
        print(f'Error: {response.status_code}')
        return None, None, None


def get_name_from_ensembl(ensembl_id):
    # Define url
    url = f'https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json'

    # Make request
    try:
        response = requests.get(url, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            data = response.json()
            try:
                return data['display_name']
            except KeyError:
                print(f'KeyError (No name) > {ensembl_id}')
                return ensembl_id
        else:
            print(f'Error: {response.status_code} > {ensembl_id}')
            return ensembl_id
    except requests.exceptions.ConnectionError:
        print('CONNECTION ERROR >', ensembl_id)
        return ensembl_id


def get_id_from_ensembl(gene_symbol, synonym_dict):
    # Define url
    url = f'https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}?content-type=application/json'

    # Make request
    try:
        response = requests.get(url, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            data = response.json()
            try:
                return data['id']
            except KeyError:
                print(f'KeyError (No ID) > {gene_symbol}')
                return '.'
        else:
            print(f'Error: {response.status_code} > {gene_symbol}')
            try:
                new_symbol = synonym_dict[gene_symbol]
                gene_id = get_id_from_ensembl(new_symbol, synonym_dict)
                print(f'Solved ({gene_symbol})')
                return gene_id
            except KeyError:
                print(f'Not solved ({gene_symbol})')
                return '.'
    except requests.exceptions.ConnectionError:
        print('CONNECTION ERROR >', gene_symbol)
        return '.'


def get_missing_id(row, synonym_dict):
    if row['ID'] == '.':
        row['ID'] = get_id_from_ensembl(row['NAME'], synonym_dict)
    return row


def write_list_to_file(fpath, list_object):
    with open(fpath, 'w', newline='\n') as f:
        for x in list_object:
            f.write(f'{x}\n')


def get_list_from_file(fpath):
    with open(fpath, 'r') as file:
        items = file.read().splitlines()
    return items


def filter_samples(fpath, sample_list):
    # Read DF (samples as index)
    df, x, x = read_df(fpath, index_col=0)

    # Filter DF
    df = df.loc[sample_list, :]

    return df


def prep_pam50(df):
    df_new = df.copy()[df['sampleID'].astype(str).str.endswith('01')]  # Drop normal samples >>> 11
    df_new['case_id'] = df_new['sampleID'].apply(get_case_from_sample)  # Get case ID
    df_new['pam50'] = df_new.loc[:, ['PAM50Call_RNAseq', 'PAM50_mRNA_nature2012']].apply(get_pam50, axis=1)  # Get PAM50
    df_new.drop(['sampleID', 'PAM50Call_RNAseq', 'PAM50_mRNA_nature2012'], inplace=True, axis=1)
    df_new['pam50'] = df_new['pam50'].fillna('unknown')
    df_new = df_new.groupby('case_id', as_index=False)['pam50'].apply(select_pam)

    return df_new


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


def filter_df(df, n, key=None, data_files=None, synonym_dict=None):
    if key == 'RNA':
        gene_names = get_list_from_file(data_files['DEG_PAPER'])
        gene_ids = [get_id_from_ensembl(x, synonym_dict) for x in gene_names]

        keep = list(set(df.columns.tolist()).intersection(gene_ids))
        keep.sort()

        df_filtered = df.loc[:, keep]
    else:  # Filter by frequency
        zero_counts = (df == 0).sum()
        top_genes = zero_counts.nsmallest(n).index
        df_filtered = df[top_genes]

    return df_filtered


def main_function():
    # DEFINE INPUT/OUTPUT
    input_dir = ''   # DIR containing downloaded files 
    output_dir = ''  # DIR to store processed files
    img_dir = ''     # DIR containing raw images
    input_aux_dir = Path(__file__).resolve().parent.parent / 'data'

    files_in_dir = os.listdir(input_dir)
    data_files = dict()
    for file in files_in_dir:
        fpath = os.path.join(input_dir, file)
        if 'CLIN_All' in file:
            data_files['CLIN'] = fpath
        elif 'Gene_level_ASCAT3' in file:
            data_files['CNV'] = fpath
        elif 'MIR' in file:
            data_files['MIR'] = fpath
        elif 'FPKM' in file:
            data_files['RNA'] = fpath
        elif 'SNP' in file:
            data_files['SNP'] = fpath
        else:
            print(f'The file {file} cannot be recognized')
            print('---------------------------')

    # DEFINE AUX FILES/DATA
    data_files['CLIN_PAM'] = os.path.join(input_aux_dir, 'XENA_TCGA_BRCA_Phenotypes.txt')
    data_files['CLIN_OT'] = os.path.join(input_aux_dir, 'brca_tcga_clinical_data_firehose.tsv')
    data_files['GENE_NAMES'] = os.path.join(input_aux_dir, 'gene_names_synonyms.csv')
    data_files['MUTSIG'] = os.path.join(input_aux_dir, 'CGN_gene_list.csv')
    data_files['DEG_PAPER'] = os.path.join(input_aux_dir, 'DEG_BRCA_Paper_2021.txt')
    data_files['IMG'] = os.path.join(input_dir, 'annotated_tcga_samples.txt')

    # GET GENE NAMES SYNONYMS
    synonym_df = pandas.read_csv(data_files['GENE_NAMES'], sep=',', header=0, index_col=0)
    synonym_dict = synonym_df['SYMBOL'].to_dict()

    # PREPROCESS DATA
    keys_preprocess = ['CLIN', 'SNP', 'CNV', 'RNA', 'MIR']
    samples_dict = dict()
    for k in keys_preprocess:
        in_file = data_files[k]
        out_file = os.path.join(output_dir, f'BRCA_{k}_all.csv')
        out_df, samples = process_tcga_data(in_file, k, append_aux=True, data_files=data_files, synonym_dict=synonym_dict)

        samples_dict[k] = samples
        out_df.to_csv(out_file, sep='\t', lineterminator='\n', index=False, header=True)

    # FILTER DATA
    keys_filter = ['SNP', 'CNV', 'RNA']
    for k in keys_filter:
        in_file = os.path.join(output_dir, f'BRCA_{k}_all.csv')
        out_file = os.path.join(output_dir, f'BRCA_{k}_filter.csv')
        out_df = filter_tcga_data(in_file, k, data_files, synonym_dict)
        out_df.to_csv(out_file, sep='\t', lineterminator='\n', index=True, header=True)  # index are case_ids

    # GET LISTS OF COMMON SAMPLES
    common_sample_list = list(set(samples_dict['CLIN']).intersection(*samples_dict.values()))
    common_sample_list.sort()
    samples_file = os.path.join(output_dir, 'BRCA_SAMPLES_common_ALL.txt')
    write_list_to_file(samples_file, common_sample_list)

    print('* Common samples (All omics) >', len(common_sample_list))

    # Filter DF to keep IMG samples
    img_sample_list = get_list_from_file(data_files['IMG'])
    img_sample_list = [get_case_from_sample(x) for x in img_sample_list]
    img_common_samples = list(set(img_sample_list).intersection(common_sample_list))
    img_common_samples.sort()
    samples_file = os.path.join(output_dir, 'BRCA_SAMPLES_common_ALL_and_IMG.txt')
    write_list_to_file(samples_file, img_common_samples)

    print('* Common samples (ALL + IMG anotada) >', len(img_common_samples))

# RUN MAIN
if __name__ == "__main__":
    main_function()
