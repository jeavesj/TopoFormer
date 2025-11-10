from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from scipy.stats import pearsonr
from transformers import BatchFeature
import torch
from datasets import Dataset
import numpy as np
import pandas as pd
import os, pickle, glob
import argparse
import sys
import argparse

from top_transformer import TopTForImageClassification
from top_transformer import TopTForPreTraining


def metrics_func(true_value, predict_value):
    # metrics
    r2 = metrics.r2_score(true_value, predict_value)
    mae = metrics.mean_absolute_error(true_value, predict_value)
    mse = metrics.mean_squared_error(true_value, predict_value)
    rmse = mse ** 0.5
    pearson_r = pearsonr(true_value, predict_value)[0]
    pearson_r2 = pearson_r ** 2

    # print
    print(f"Metric - r2: {r2:.3f} mae: {mae:.3f} mse: {mse:.3f} "
          f"rmse: {rmse:.3f} pearsonr: {pearson_r:.3f} pearsonr2: {pearson_r2:.3f}")
    return r2, mae, mse, rmse, pearson_r, pearson_r2


def get_predictions(
    top_feature_array = None,
    test_label_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/downstream_task_labels/CASF2007_core_test_label.csv',
    scaler_path = r'/home/chendo11/workfolder/TopTransformer/code_pkg/pretrain_data_standard_minmax_6channel_large.sav',
    model_path = r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_regression_CASF2007/selected_global_para_32_0.0001_from_3w/model_cls_10000_0',
    save_path = r'/home/chendo11/workfolder/TopTransformer/Output_dir/for_consensus_predict_deep_result_20times/scoring_casf2007/para_32_0.0001.npy'
):

    # data prepare and preprocess
    label_df = pd.read_csv(test_label_file, header=0, index_col=0)
    scaler = pickle.load(open(scaler_path, 'rb'))
    num_sample, num_channel, height, width = np.shape(top_feature_array)
    data_0 = np.reshape(top_feature_array, [num_sample, num_channel*height*width])
    scaled_data = scaler.transform(data_0).reshape([num_sample, num_channel, height, width])
    model_inputs = BatchFeature({"topological_features": scaled_data}, tensor_type='pt')

    # load model
    model = TopTForImageClassification.from_pretrained(model_path)

    # prediction
    with torch.no_grad():
        outputs = model(**model_inputs)
    predicted_value = outputs.logits.squeeze().numpy()
    result_dict = {"predict": predicted_value, "true": label_df.values.squeeze(1)}

    # save result
    np.save(save_path, result_dict, allow_pickle=True)

    # optional, metric
    # metrics_func(result_dict['predict'], result_dict['true'])
    return result_dict


def main_get_predictions_for_scoring():
    def get_top_feature_from_dict(top_feature_file, test_label_file):
        top_feature_dict = np.load(top_feature_file, allow_pickle=True).item()
        label_df = pd.read_csv(test_label_file, header=0, index_col=0)
        top_feature_array = [np.float32(top_feature_dict[key]) for key in label_df.index.tolist()]
        return top_feature_array

    top_feature_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/all_feature_ele_scheme_1-lap0_rips_12-6channel-212-filtration50.npy'
    test_label_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/downstream_task_labels/CASF2016_core_test_label.csv'
    scaler_path = r'/home/chendo11/workfolder/TopTransformer/code_pkg/pretrain_data_standard_minmax_6channel_filtration50-12.sav'
    model_pathes = glob.glob(r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_regression_CASF2016_filtration50_212_12/selected_global_para_64_0.00008_from_3w/model_cls_64_0.00008_*')
    for i, model_p in enumerate(model_pathes):
        print(i, model_p)
        if not os.path.exists(os.path.join(model_p, 'all_results.json')):
            continue

        save_path = rf'/home/chendo11/workfolder/TopTransformer/Output_dir/for_consensus_predict_deep_result_20times/scoring_casf2016_212_50/model_para_64_0.00008_{i}.npy'

        get_predictions(
            top_feature_array=get_top_feature_from_dict(top_feature_file, test_label_file),
            test_label_file=test_label_file,
            scaler_path=scaler_path,
            model_path=model_p,
            save_path=save_path,
        )

    return None


def main_get_predictions_for_scoring_2020():
    def get_top_feature_from_dict(top_feature_file, test_label_file):
        top_feature_dict = np.load(top_feature_file, allow_pickle=True).item()
        label_df = pd.read_csv(test_label_file, header=0, index_col=0)
        top_feature_array = [np.float32(top_feature_dict[key]) for key in label_df.index.tolist()]
        return top_feature_array

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_file', type=str, default='crystal_feats.npy', help='Input feature file path')
    parser.add_argument('--label_file', type=str, default='labels.csv', help='Input test set labels filepath')
    parser.add_argument('--scaler_file', type=str, default='code_pkg/pretrain_data_standard_minmax_6channel_large.sav', help='Scaler path')
    parser.add_argument('--model_paths', type=str, default='saved_model_last_4w', help='Model directory path')
    parser.add_argument('--outdir', type=str, default='preds', help='Directory path for storing output predictions')
    args = parser.parse_args()
    
    top_feature_file = args.feature_file
    test_label_file = args.label_file
    scaler_path = args.scaler_file
    model_pathes = glob.glob(args.model_paths)
    for i, model_p in enumerate(model_pathes):
        print(i, model_p)
        if not os.path.exists(os.path.join(model_p, 'all_results.json')):
            continue

        save_path = os.path.join(args.outdir, f'model_{i}.npy')

        get_predictions(
            top_feature_array=get_top_feature_from_dict(top_feature_file, test_label_file),
            test_label_file=test_label_file,
            scaler_path=scaler_path,
            model_path=model_p,
            save_path=save_path,
        )
    return None


def get_predictions_for_docking(feature_dir, scaler_path, model_dir, label_file, out_dir):
    
    os.makedirs(out_dir, exist_ok=True)
    
    label_df = pd.read_csv(label_file, header=0, index_col=0)
    ids = label_df.index.tolist()

    # concatenating input data for individual structures into single, 4D array
    features = []
    for sample_id in ids:
        npy_path = os.path.join(feature_dir, f'{sample_id}.npy')
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f'Missing feature file for ID "{sample_id}": {npy_path}')

        arr = np.load(npy_path, allow_pickle=True)
        if arr.ndim != 3:
            raise ValueError(f'Feature file {npy_path} has incorrect shape {arr.shape}.'
                            f'Expected shape (6, X, 143).')
        
        arr = arr[:,0::2, :]
        features.append(arr.astype('float32'))
    
    # get docking predicted result for all
    top_feature_array = np.stack(features, axis=0)

    save_path = os.path.join(out_dir, 'docking_predictions.npy')

    results_dict = get_predictions(
        top_feature_array=top_feature_array,
        test_label_file=label_file,
        scaler_path=scaler_path,
        model_path=model_dir,
        save_path=save_path,
    )
    
    print(results_dict)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function',  type=str, default='docking', help='Specify which function to perform. \
                        Options [docking, scoring, scoring_2020] correspond to functions main_get_predictions_for_docking(), \
                        main_get_predictions_for_scoring(), and main_get_predictions_for_scoring_2020(), respectively.')
    parser.add_argument('--feature_dir', type=str, required=True, help='Path to directory containing featurized data as .npy files.')
    parser.add_argument('--scaler_path', type=str, default='code_pkg/pretrain_data_standard_minmax_6channel_filtration50-12.sav')
    parser.add_argument('--model_dir', type=str, default='saved_model_last_4w', help='Path to pretrained model dir')
    parser.add_argument('--label_file', type=str, required=True, help='Path to label file with column 0 as id (e.g., pdbid) and column 1 as binding_energy.')
    parser.add_argument('-o', '--outdir', type=str, default='outputs', help='Path to output directory.')
    args = parser.parse_args()
    
    if args.function == 'docking':
        get_predictions_for_docking(args.feature_dir, args.scaler_path, args.model_dir, args.label_file, args.outdir)
    elif args.function == 'scoring':
        main_get_predictions_for_scoring()
    elif args.function == 'scoring_2020':
        main_get_predictions_for_scoring_2020()
    else:
        raise RuntimeError(f'Unsupported input for --function: {args.function}. Supported options are "docking", "scoring", and "scoring_2020"')


if __name__ == "__main__":
    main()
    print('End!')