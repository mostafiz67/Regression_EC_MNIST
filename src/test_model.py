"""
Author: Md Mostafizur Rahman
File: Train and Test the model (including ECs calculations)
"""

import os, torch
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List
from warnings import filterwarnings
from numpy import ndarray
from torch.utils.data import DataLoader

from src import data_preprocess, my_model, train_model
from src.constants import CHECKPOINT, NB_REP, OUT, K_Fold

ECMethod = ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", 
            "intersection_union_sample", "intersection_union_all", "intersection_union_distance"]



def regression_ec(residuals: List[ndarray], method: ECMethod) -> List[ndarray]:
    filterwarnings("ignore", "invalid value encountered in true_divide", category=RuntimeWarning)
    consistencies = []
    for pair in combinations(residuals, 2):
        r1, r2 = pair
        r = np.vstack(pair)
        sign = np.sign(np.array(r1) * np.array(r2))
        if method == "ratio-signed":
            consistency = np.multiply(sign, np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0))
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio":
            consistency = np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0)
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio-diff-signed":
            consistency = np.multiply(sign, (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2)))
            consistency[np.isnan(consistency)] = 0
        elif method == "ratio-diff":
            consistency = (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2))
            consistency[np.isnan(consistency)] = 0
        elif method =="intersection_union_sample":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choice_numerator = [np.minimum(r1, r2), np.minimum(np.abs(r1), np.abs(r2))]
            choice_denominator = [np.maximum(r1, r2), np.maximum(np.abs(r1), np.abs(r2))]
            numerator = np.select(conditions, choice_numerator, np.zeros(len(r1)))
            denominator = np.select(conditions, choice_denominator, np.abs(np.add(np.abs(r1), np.abs(r2))))
            consistency = np.divide(numerator, denominator)
            consistency[np.isnan(consistency)] = 1
        elif method =="intersection_union_all":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choice_numerator = [np.minimum(r1, r2), np.minimum(np.abs(r1), np.abs(r2))]
            choice_denominator = [np.maximum(r1, r2), np.maximum(np.abs(r1), np.abs(r2))]
            numerator = np.select(conditions, choice_numerator, np.zeros(len(r1)))
            denominator = np.select(conditions, choice_denominator, np.abs(np.add(np.abs(r1), np.abs(r2))))
            consistency = np.divide(np.sum(numerator), np.sum(denominator)) # all sum and then divide
            consistency = np.nan_to_num(consistency, copy=True, nan=1.0)
        elif method =="intersection_union_distance":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choiceValue = [np.abs(np.subtract(np.abs(r1), np.abs(r2))), np.abs(np.subtract(np.abs(r1), np.abs(r2)))]
            consistency = np.select(conditions, choiceValue, np.add(np.abs(r1), np.abs(r2)))
        else:
            raise ValueError("Invalid method")
        consistencies.append(consistency)
    filterwarnings("default", "invalid value encountered in true_divide", category=RuntimeWarning)
    return consistencies


def calculate_consitencies(rep_residuals, rep_gofs):
    summaries = []
    for method in ECMethod:
        consistencies: ndarray = np.array(regression_ec(list(rep_residuals), method))
        summaries.append(
                pd.DataFrame(
                    {
                        "Method": method,
                        "EC": consistencies.mean(),
                        "EC_vec_sd": consistencies.std(ddof=1) if method == "intersection_union_all" else consistencies.mean(axis=0).std(ddof=1),
                        "EC_scalar_sd": "NA" if method == "intersection_union_all" else consistencies.mean(axis=1).std(ddof=1),
                        "Test_MAE": np.mean(rep_gofs["Test_MAE"]),
                        "Test_MAE_sd": np.std(rep_gofs["Test_MAE"], ddof=1),
                        "Test_MAPE": np.mean(rep_gofs["Test_MAPE"]),
                        "Test_MAPE_sd": np.std(rep_gofs["Test_MAPE"], ddof=1),
                        "Test_MSqE": np.mean(rep_gofs["Test_MSqE"]),
                        "Test_MSqE_sd": np.std(rep_gofs["Test_MSqE"], ddof=1),
                        "Test_R2": np.mean(rep_gofs["Test_R2"]),
                        "Test_R2_sd": np.std(rep_gofs["Test_R2"], ddof=1),
                    },
                    index=[0],
                )
        )
        summary = pd.concat(summaries, axis=0, ignore_index=True)
    return summary


def calculate_ECs():
    rep_residuals, rep_gofs, rep_actual_lab, rep_predict_y = [], [], [], []
    for i in range(NB_REP):
        print("-------------------------Repetition Number---------------", i)
        fold_gofs, fold_residuals, fld_act_lab, fld_pred_y = train_model.model_train(i) # training and saving model
        rep_residuals.append(fold_residuals)
        rep_actual_lab.append(fld_act_lab)
        rep_predict_y.append(fld_pred_y)
        rep_gofs.append(fold_gofs)
    final_rep_gofs = pd.concat(rep_gofs, axis=0, ignore_index=True)
    # print(np.shape(rep_residuals), np.shape(rep_actual_lab), np.shape(rep_predict_y), 
    # np.shape(np.array(rep_residuals).reshape(250,-1)))
    rep_residuals = np.array(rep_residuals).reshape(NB_REP*K_Fold,-1)  # rep*folds

    # #saving residuals
    # all_residuals_csv = pd.DataFrame(rep_residuals)
    # residual_csv_file_name = 'residual_csv_file.csv'
    # outfile_residual = OUT / residual_csv_file_name
    # all_residuals_csv.to_csv(outfile_residual, index=False, header=False)

    # #saving actual
    # all_actual_lab_csv = pd.DataFrame(rep_actual_lab)
    # actual_lab_csv_file_name = 'actual_lab_csv_file.csv'
    # outfile_actal_lab = OUT / actual_lab_csv_file_name
    # all_actual_lab_csv.to_csv(outfile_actal_lab, index=False, header=False)

    # #saving prediction
    # all_actual_lab_csv = pd.DataFrame(rep_predict_y)
    # actual_lab_csv_file_name = 'predict_y_csv_file.csv'
    # outfile_actal_lab = OUT / actual_lab_csv_file_name
    # all_actual_lab_csv.to_csv(outfile_actal_lab, index=False, header=False)

    final_output= calculate_consitencies(rep_residuals, final_rep_gofs)
    print(final_output)
    filename = "error.csv"
    outfile = OUT / filename
    final_output.to_csv(outfile)
    print(f"Saved results for error to {outfile}")


def preds_actual_compare():
    for rep in range(0, NB_REP):
        for fold_num in range(0,K_Fold):
            model = my_model.get_model() #Loading Model
            model_filename = "baseline_" + str(rep) + "_" + str(fold_num) + ".h5"
            model.load_state_dict(torch.load(os.path.join(CHECKPOINT, model_filename)))
            _, test_dataloader = data_preprocess.data_loaders() #loading data
            data_loader = test_dataloader = DataLoader(test_dataloader, batch_size=10000)
            sample = next(iter(data_loader))
            imgs, lbls = sample
            actual_number = lbls[:10].detach().numpy()

            test_output, _ = model(imgs[:10])
            # print(test_output)
            preds_test = test_output.squeeze().detach().numpy()
            # print(np.shape(preds_test), np.shape(actual_number))
            print(f'Prediction Number for Rep: {rep} and K_Fold {fold_num}: {preds_test}')
            print(f'Actual Number  for Rep: {rep} and K_Fold {fold_num}: {actual_number}')

if __name__ == "__main__":
    calculate_ECs()