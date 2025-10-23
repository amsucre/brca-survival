import pandas as pd


# ===== AUX FUNCTIONS
def get_list_from_file(fpath):
    with open(fpath, 'r') as file:
        items = file.read().splitlines()
    return items


def _main_(in_file, out_file, sample_file=None, filter_idc=False, sep='\t'):
    print('Input:')
    print('*', in_file)
    print('*', sample_file)
    print('* filter IDC:', filter_idc)

    n_bins = 4                                  # number of discretized survival bins
    patient_strat = True                        # patient-level stratificaction (no se usa?)
    label_col = 'survival_months'               # name of the column with the survival time
    eps = 1e-6

    patient_data = pd.read_csv(in_file, low_memory=False, sep=sep)      # prevent Pandas from guessing data types incorrectly

    # Make sure each sample is linked to a patient
    if 'case_id' not in patient_data:
        patient_data.index = patient_data.index.str[:12]
        patient_data['case_id'] = patient_data.index
        patient_data = patient_data.reset_index(drop=True)

    # Set survival labels
    if not label_col:
        label_col = 'survival_months'
    else:
        assert label_col in patient_data.columns

    # Filter only IDC Cases (most common subtype in the dataset, ensures homogeneous survival patterns)
    if filter_idc:
        if "IDC" in patient_data['oncotree_code'].tolist():  # no entraba nunca sin el "tolist()"
            patient_data = patient_data[patient_data['oncotree_code'] == 'IDC']

    # If a subset of samples is given, keep only those samples before binning
    if sample_file:
        sample_list = get_list_from_file(sample_file)
        patient_data = patient_data.loc[patient_data['case_id'].isin(sample_list), :]

    # Extract unique patients for binning survival times
    unique_patients_df = patient_data.drop_duplicates(['case_id']).copy()

    uncensored_df = unique_patients_df[unique_patients_df['censorship'] < 1]      # keep only uncensored patients (experienced the event), the binning of survival times is done only on patients with known survival times

    # Discretize Survival Time into Bins
    disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)

    # Ensure Binning Covers Full Survival Range
    q_bins[-1] = patient_data[label_col].max() + eps
    q_bins[0] = patient_data[label_col].min() - eps

    # Assign Discretized Labels to Patients
    disc_labels, q_bins = pd.cut(unique_patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)     # assign a bin (0,1,2,3) to each patient survival time
    unique_patients_df.insert(2, 'label', disc_labels.values.astype(int))

    # Create a mapping from case_id to assigned label
    patient_label_mapping = unique_patients_df[['case_id', 'label']]

    # Merge the labels back to the full dataset, keeping all slides
    patient_data = patient_data.merge(patient_label_mapping, on='case_id', how='left')

    # Mapping from (survival_bin, censoring_status) to unique class index
    label_dict = {}
    key_count = 0
    for i in range(len(q_bins)-1):
        for c in [0, 1]:
            print('{} : {}'.format((i, c), key_count))
            label_dict.update({(i, c): key_count})
            key_count += 1

    # Assign final labels (from 0 to 7 instead of the four bins)
    for i in patient_data.index:
        key = patient_data.loc[i, 'label']
        patient_data.at[i, 'disc_label'] = int(key)
        censorship = patient_data.loc[i, 'censorship']
        key = (key, int(censorship))
        patient_data.at[i, 'label'] = label_dict[key]

    # Ensure disc_label is integer
    patient_data['disc_label'] = patient_data['disc_label'].astype(int)

    # Save final dataset with all slides and labels
    patient_data.to_csv(out_file, index=False, sep=sep)
    print(f"Saved dataset with labels to {out_file}")


# ========== MAIN
if __name__ == '__main__':
    # All available samples
    input_path = ''   # Tabular data file
    output_path = ''  # Output file
    _main_(input_path, output_path)
