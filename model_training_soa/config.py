from dataclasses import dataclass
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainConfig(object):

    eager_mode = False
    output_path = Path(REPO_DIR, 'data', 'output', 'segmentation', 'model')  # Path to store the trained model
    splits_dir = Path(REPO_DIR, 'data', 'data_splits', 'splits_segmentation')
    splits_dir_survival = Path(REPO_DIR, 'data', 'data_splits', 'tcga_brca_final')
    csv_data_path = Path(REPO_DIR, 'data', 'combined.csv')
    full_csv_data_path = Path(REPO_DIR, 'data', 'tcga_MGCT_PORPOISE.csv')
    features_dir = Path(REPO_DIR, 'data', 'patches_TCGA_2048', 'features_uni')
    # features_dir = Path(REPO_DIR, 'data', 'features_uni')
    # features_dir = Path(REPO_DIR, 'data', 'FEATURES_DIRECTORY_RESNET50')
    signatures_path = Path(REPO_DIR, 'data', 'MGCT', 'dataset_csv', 'signatures.csv')
    inference_path = Path('')  # DEFINE: raw data directory
    results_dir = Path(REPO_DIR, 'data', 'RESULTS_TRAINING_UNI_TMP')
    model_name = 'test_model'

    mlflow = True
    mlflow_server_tracking_uri = ''  # DEFINE: URL to MLflow server
    mlflow_s3_endpoint_url = ''  # DEFINE: URL to MLflow backend store
    experiment_name = ''  # DEFINE: mlflow experiment name

    learning_rate = 1e-4  # Learning rate value
    scheduler = False
    scheduler_epochs = 50
    batch_size = 8  # Batch size value 8,16,32
    number_of_epochs = 80  # Number of epochs (recommended > 1000)
    seed = 1234  # Seed value
    early_stopping = 20  # Number of epochs in which val_loss is not improving to stop training
    loss = 'dice'
    num_classes = 4
    filter_classes = True
    filter_classes_binary = False
    tils = False
    class_weights = [1, 1, 1]
    mpp = 0.5

    img_size = 512

    dataset_train_path = Path(REPO_DIR, 'data', 'tfrecords', 'train*.tfrecord')
    dataset_val_path = Path(REPO_DIR, 'data', 'tfrecords', 'val*.tfrecord')
    dataset_test_path = Path(REPO_DIR, 'data', 'tfrecords', 'test*.tfrecord')

@dataclass(frozen=True)
class EvaluateConfig(object):
    csv_data_path = Path(REPO_DIR, 'data', 'tcga_MGCT_PORPOISE.csv')
    splits_dir_survival = Path(REPO_DIR, 'data', 'data_splits', 'tcga_brca_final')
    features_dir = Path(REPO_DIR, 'data', 'FEATURES_DIRECTORY_RESNET50')
    # features_dir = Path(REPO_DIR, 'data', 'features_convnext')
    # results_dir_evaluation = Path(REPO_DIR, 'data', 'RESULTS_TRAINING_FINAL', '5foldcv',
    #                               'MGCT_s1_s2_ffn_linear_snn_mha_sig_concat',
    #                               'tcga_brca_MGCT_s1_s2_ffn_linear_snn_mha_sig_concat_s1')
    results_dir_evaluation = Path(REPO_DIR, 'data', 'RESULTS_TRAINING_FINAL', '5foldcv', 'MCAT_sig_concat',
                                 'tcga_brca_MCAT_sig_concat_s1')

@dataclass(frozen=True)
class InferenceConfig(object):
    slides_path = Path(REPO_DIR, 'data', 'raw_data')
    processed_slides_path = Path(REPO_DIR, 'data', 'patches_TCGA_2048', 'patches')
    save_csv_path = Path(REPO_DIR, 'data')

@dataclass(frozen=True)
class DataConfig(object):
    input_path = Path(REPO_DIR, 'data', 'raw_data', 'images')
    output_path = Path(REPO_DIR, 'data', 'tfrecords')


@dataclass(frozen=True)
class FeatureExtractionConfig(object):
    # data_h5_dir = Path(REPO_DIR, 'data', 'patches_TCGA_2048')
    # data_slide_dir = Path(REPO_DIR, 'data', 'raw_data')
    # csv_path = Path(REPO_DIR, 'data', 'patches_TCGA_2048', 'process_list_autogen.csv')
    # feat_dir = Path(REPO_DIR, 'data', 'patches_TCGA_2048', 'features_resnet')
    data_h5_dir = Path(REPO_DIR, 'data', 'DHMC', 'patches')
    data_slide_dir = Path(REPO_DIR, 'data', 'DHMC', 'raw_data')
    csv_path = Path(REPO_DIR, 'data', 'DHMC', 'patches', 'process_list_autogen.csv')
    feat_dir = Path(REPO_DIR, 'data', 'DHMC', 'features_uni')
    model_checkpoint = Path(REPO_DIR, 'data', 'checkpoints', 'uni', 'vit_large_patch16_224.dinov2.uni_mass100k', 'pytorch_model.bin')

@dataclass(frozen=True)
class FeatureExtractionConfigOurs(object):
    source_dir = Path(REPO_DIR, 'data', 'DHMC', 'raw_data')
    save_dir = Path(REPO_DIR, 'data', 'DHMC', 'patches')
    csv_path = Path(REPO_DIR, 'data', 'patches_TCGA_2048', 'process_list_autogen.csv')
    feat_dir = Path(REPO_DIR, 'data', 'patches_TCGA_2048', 'features_uni')