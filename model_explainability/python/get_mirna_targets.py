import ast
from collections import Counter
from pathlib import Path

import pandas


# =========== AUX FUNCTIONS
def get_mirna_targets(src_name: str, in_path: Path, info: dict, mir_list: list, gene_path: Path):
    print(f'\n==========\nProcessing {src_name} file ({in_path.name})...')

    # Read data
    cols = [info['col_mir'], info['col_gene'], info['col_filter']]
    df = pandas.read_csv(in_path, sep=info['sep'], header=info['header'], usecols=cols)
    df = df.loc[:, cols]  # Make sure order is correct
    df.columns = ['MIRNA', 'GENE', 'TO_FILTER']
    df['MIRNA'] = df['MIRNA'].str.lower()

    if info['filter_class'] == 'in':
        df['TO_FILTER'] = df['TO_FILTER'].str.lower()

    print('* Initial size:', df.shape)

    # Keep miRNA of interest
    final_list = mir_list + [f'{x}-3p' for x in mir_list] + [f'{x}-5p' for x in mir_list]
    df_list = df['MIRNA'].tolist()
    final_list = list(set(df_list).intersection(set(final_list)))

    print('* Found miRNA:', len(final_list))

    df = df.loc[df['MIRNA'].isin(final_list), :]

    print('* Size after filter miRNA:', df.shape)

    # Filter by given condition
    if info['filter_class'] == '>=':
        df = df.loc[df['TO_FILTER'] >= info['filter_thr'], :]
        print('* Min value:', df['TO_FILTER'].min())
        print('* Max value:', df['TO_FILTER'].max())
    elif info['filter_class'] == '<':
        df = df.loc[df['TO_FILTER'] < info['filter_thr'], :]
        print('* Min value:', df['TO_FILTER'].min())
        print('* Max value:', df['TO_FILTER'].max())
    elif info['filter_class'] == 'in':
        df = df.loc[df['TO_FILTER'].isin(info['filter_thr']), :]
        print('* Kept values:', set(df['TO_FILTER'].tolist()))
    else:  # no filter
        df = df.copy()

    print('* Size after filter by thershold:', df.shape)

    # If requested, get gene symbol (when ENS ID is given)
    if info['get_symbol']:
        print('* Obtaining missing gene symbols...')
        df['GENE'] = get_gene_symbol(df['GENE'], gene_path)

    # Drop "added" -3p, -5p, so all genes matched to the same "main" miRNA are merged together
    df['MIRNA'] = df['MIRNA'].str.replace(r"(-3p|-5p)$", '', regex=True)

    # Drop matching duplicates
    df.drop_duplicates(subset=['MIRNA', 'GENE'], inplace=True)
    df.sort_values(by='MIRNA', inplace=True)

    print('* Size after drop duplicates:', df.shape)

    # Merge all matches for a given miRNA
    df_grouped = df.groupby('MIRNA')['GENE'].agg(list).to_frame()

    print('* Size after group:', df_grouped.shape)

    return df_grouped


def get_gene_symbol(gene_serie: pandas.Series, gene_path: Path):
    gene_list = gene_serie.tolist()
    chunks = pandas.read_csv(gene_path, sep='\t', header=0, index_col=0, chunksize=1000000)
    kept_n = 0
    kept_df = None
    for chunk in chunks:
        chunk_keep = chunk[chunk.index.isin(gene_list)]
        kept_n += chunk_keep.shape[0]

        if kept_df is None:
            kept_df = chunk_keep.copy()
        else:
            kept_df = pandas.concat([kept_df, chunk_keep.copy()])

        if kept_n == len(gene_list):
            print('All matches found')
            break

    annot_df = pandas.merge(gene_serie, kept_df, left_on=['GENE'], right_index=True, how='left')

    return annot_df['SYMBOL']


def get_final_gene_list(row: pandas.Series, min_rep: int = 2):
    # Get flatten gene list
    all_genes = [gene for gene_list in row for gene in gene_list]

    # Count gene occurrences
    gene_counts = Counter(all_genes)

    # Filter genes that appear in at least min_rep
    final_genes = [gene for gene, count in gene_counts.items() if count >= min_rep]

    return final_genes


def get_relevant_values(in_path: Path, colname: str, sep: str, contain_list: list, exclude_list: list = None):
    print(f'\nProcessing {in_path.name} to get relevant experiments...')
    init_list = pandas.read_csv(in_path, sep=sep, usecols=[colname])[colname].tolist()
    init_list = [x.lower() for x in set(init_list)]

    print('* Initial size:', len(init_list))

    final_list = []
    for x in contain_list:
        final_list += [y for y in init_list if x in y]
    final_list = list(set(final_list))

    if exclude_list:
        for x in exclude_list:
            try:
                final_list.remove(x)
            except ValueError:
                pass

    print('* Final size:', len(final_list), final_list)

    return final_list


def prep_gene_file(f_in: Path, f_out: Path):
    df_chunks = pandas.read_csv(f_in, sep='\t', header=0, usecols=['RNA_nucleotide_accession.version', 'Symbol'],
                                na_values=['-'], chunksize=50000)

    new = True
    i = 1
    for chunk in df_chunks:
        chunk.columns = ['RNA_ACCESSION', 'SYMBOL']

        chunk.dropna(inplace=True)  # en cualquiera de las 2 columnas

        if chunk.empty:
            print(f'Chunk {i}: Dropped')
            i += 1
            continue

        mask = chunk['RNA_ACCESSION'].str.startswith(('NM', 'XM'))
        chunk = chunk[mask]

        chunk['RNA_ACCESSION'] = chunk['RNA_ACCESSION'].str.split('.').str[0]

        chunk.drop_duplicates(inplace=True)

        if new:
            chunk.to_csv(f_out, sep='\t', header=True, index=False, mode='w')
            new = False
        else:
            chunk.to_csv(f_out, sep='\t', header=False, index=False, mode='a')

        print(f'Chunk {i}:', chunk.shape)

        i += 1


# =========== MAIN
if __name__ == "__main__":
    # # Prep gene file (RUN ONLY ONCE)
    gene_in = Path('')    # gene2refseq file
    gene_out = Path('')   # output path (gene2refseq filtered)
    prep_gene_file(gene_in, gene_out)


    # # Define input
    parent_dir = Path(__file__).resolve().parent.parent  # gets "model_explainability" path
    in_dir = Path(parent_dir / 'data' / 'miRNA_ANNOT')   # Directory containing miRNA annotations (downloaded from annotation official sources)
    shap_dir = Path('') # Directory containing SHAP analysis results
    shap_out = Path('')  # Directory where to store new results
    feature_file = shap_dir / 'shap_top_features.txt'

    src_list = {'TargetScan': in_dir / 'TargetScan_V8' / 'Predicted_Targets_Context_Scores.default_predictions.txt',
                'miRDB': in_dir / 'miRDB_V6' / 'miRDB_v6.0_prediction_result.txt',
                'miRTarBase': in_dir / 'miRTarBase_V10' / 'hsa_MTI.csv',
                'TarBase': in_dir / 'TarBase_V9' / 'Homo_sapiens.tsv',
                'miRWalk': in_dir / 'miRWalk_V3' / 'hsa_miRWalk_3UTR.txt'
                }

    # # Define list of relevant experiment types for filtering porpoises
    # # # mirTarBase >> Keep options including Reporter Assay, Western bolt, qPCR (stronger evidence)
    keep_list = ['reporter assay', 'qrt-pcr', 'western blot']
    exp_mtb = get_relevant_values(src_list['miRTarBase'], 'Experiments', ',', keep_list)
    
    # # # TarBase >> Keep options including Reporter Assay, Western bolt, qPCR (stronger evidence)
    keep_list = ['reporter assay', 'qpcr', 'western blot']  # , 'clash', 'clip'] >> recomienda gemini, pero no MTB
    exp_tb = get_relevant_values(src_list['TarBase'], 'experimental_method', '\t', keep_list, ['biotin-qpcr'])
    
    # # Define processing parameters per source
    src_info = {'TargetScan': {'sep': '\t', 'header': 0, 'col_mir': 'miRNA', 'col_gene': 'Gene Symbol',
                               'col_filter': 'context++ score', 'filter_class': '<', 'filter_thr': -0.1, 'get_symbol': False}, 
                'miRDB': {'sep': '\t', 'header': None, 'col_mir': 0, 'col_gene': 1,
                          'col_filter': 2, 'filter_class': '>=', 'filter_thr': 85, 'get_symbol': True},
                'miRTarBase': {'sep': ',', 'header': 0, 'col_mir': 'miRNA', 'col_gene': 'Target Gene',
                               'col_filter': 'Experiments', 'filter_class': 'in', 'filter_thr': exp_mtb, 'get_symbol': False},
                'TarBase': {'sep': '\t', 'header': 0, 'col_mir': 'mirna_name', 'col_gene': 'gene_name',
                            'col_filter': 'experimental_method', 'filter_class': 'in', 'filter_thr': exp_tb, 'get_symbol': False},
                'miRWalk': {'sep': '\t', 'header': 0, 'col_mir': 'miRNA', 'col_gene': 'Genesymbol',
                            'col_filter': 'binding_probability', 'filter_class': '>=', 'filter_thr': 0.9, 'get_symbol': False}}
    
    # # Get top miRNA
    top_mir = pandas.read_csv(feature_file, sep='\t', header=0, usecols=['MIR-D'], nrows=10)['MIR-D'].tolist()
    
    # # Get miRNA targets per SRC
    merged_df = None
    for name, path in src_list.items():
    
        src_df = get_mirna_targets(name, path, src_info[name], top_mir, gene_out)
        src_df.columns = [name]
    
        if merged_df is None:
            merged_df = src_df.copy()
        else:
            merged_df = pandas.merge(merged_df, src_df, left_index=True, right_index=True, how='outer')
    
        print('\n* Merged:')
        print(merged_df)
    
        # break
    
    # Store in case something fails afterwards. Avoid repeating all process
    merged_df.to_csv(shap_out / 'aux_mirna_gene_targets_V2.tsv', sep='\t', index=True, header=True)
    
    merged_df = pandas.read_csv(shap_out / 'aux_mirna_gene_targets_V2.tsv', sep='\t', header=0, index_col=0)
    merged_df = merged_df.map(lambda x: ast.literal_eval(x) if pandas.notna(x) else x)
    
    # Convert NAs to empty lists to avoid future errors
    merged_df = merged_df.map(lambda x: [] if str(x) == 'nan' else x)
    
    # # Keep genes recommended by at least 2 different sources
    merged_df['GENES'] = merged_df.apply(get_final_gene_list, axis=1, min_rep=3)
    merged_df['COUNT'] = merged_df['GENES'].apply(lambda x: len(x))
    merged_df = merged_df.loc[:, ['COUNT', 'GENES']]
    
    print(merged_df)
    
    # Save lists of miRNA per gene
    merged_path = shap_out / 'gene_targets_per_mirna_V2_3src.tsv'
    merged_df.to_csv(merged_path, sep='\t', header=True, index=True)
    
    # Save list of genes (global)
    final_genes = list(set([gene for gene_list in merged_df['GENES'].tolist() for gene in gene_list]))
    print('* Final gene list (n)', len(final_genes))
    gene_path = shap_out / 'MIR_top10_target_genes_V2_3src.tsv'
    with open(gene_path, 'w') as f:
        for gene in final_genes:
            f.write(gene + '\n')
    
    # AUX: Get RNA universe (all genes considered in the analysis)
    rna_in = Path('')   # Input tabular data matrix file containing list of all RNA considered (file header) 
    with open(rna_in, 'r') as file:
        first_line = file.readline()
        first_line = first_line.strip()
        first_line_elements = first_line.split('\t')
    
        rna_genes = [x for x in first_line_elements if x not in ['case_id', 'MONTHS', 'IS_CENSORED', 'OUTCOME']]
    
    rna_out = shap_out / 'RNA_universe.tsv'
    with open(rna_out, 'w') as f:
        for gene in rna_genes:
            f.write(gene + '\n')
