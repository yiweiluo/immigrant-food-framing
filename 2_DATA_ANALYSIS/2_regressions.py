#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm, trange
from collections import Counter
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf

def load_reviews_df(path_to_reviews_df, debug):
    print("\nReading in reviews_df...")
    if False:#debug:
        print(f"\nDebug mode ON; limiting to first 5000 lines...")
    else:
        print(f"\nDebug mode OFF; reading in entire dataframe...")
        reviews_df = pd.read_pickle(path_to_reviews_df)
    reviews_df['biz_macro_region'] = reviews_df['biz_cuisine_region'].apply(lambda x: 'us' if x == 'us' else 'non-us')
    reviews_df['biz_cuisines'] = reviews_df['biz_cuisines'].apply(lambda x: list(x))

    print(f"\tDone! Read in df with shape {reviews_df.shape}.")
    print(reviews_df.head())
    print()
    print(reviews_df['biz_macro_region'].value_counts())
    print()
    print(reviews_df['biz_cuisine_region'].value_counts())
    print()
    print(reviews_df[['biz_median_nb_income','biz_nb_diversity']].describe())
    
    return reviews_df

def zscore_df(df, anchor='agg'):
    print(f"\nz-scoring non-dummy variables...")
    
    for feat in ["review_len","biz_mean_star_rating","biz_median_nb_income","biz_nb_diversity"]:
        df[f"{feat}"] = stats.zscore(df[feat])

    dep_vars = [f"{var}_{anchor}_score" 
                for var in ['exotic_words','auth_words','auth_simple_words','auth_other_words','typic_words',
                            'filtered_liwc_posemo','luxury_words',
                            'hygiene_words','hygiene.pos_words','hygiene.neg_words',
                            'cheapness_words','cheapness_exp_words','cheapness_cheap_words']]
    for dep_var in dep_vars:
        df[f"{dep_var.replace('.','_')}"] = stats.zscore(df[dep_var])
    print("\tDone!")
    
    return df

def check_VIF(df):
    print("\nChecking VIF scores for multicollinearity...")
    inds = ['review_len','biz_mean_star_rating','biz_median_nb_income','biz_nb_diversity'] + \
            [x for x in df.columns if x.endswith('agg_score') 
             and 'hygiene_words' not in x 
             and 'cheapness_words' not in x
             and 'auth_words' not in x
             and '.' not in x]
    X = df[inds]

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                              for i in range(len(X.columns))]
    print(vif_data)
    print("\tDone!")
    
def get_abs_coeffs(res, ref='us'):
    intercept_row = res.params.filter(like='Intercept', axis=0)
    intercept_row.index = [ref]
    cuisine_coeffs = res.params.filter(like='_region', axis=0)
    cuisine_coeffs.index = [x.split('T.')[-1].replace(']','') for x in cuisine_coeffs.index]
    cuisine_coeffs = cuisine_coeffs + intercept_row[ref]
    cuisine_coeffs = cuisine_coeffs.append(intercept_row)
    return cuisine_coeffs

def get_abs_per_cuisine_coeffs(res):
    intercept = res.params[0]
    cuisine_coeffs = res.params.filter(like='biz_cuisine',axis=0)
    cuisine_coeffs.index = [x.replace('biz_cuisine_','') for x in cuisine_coeffs.index]
    cuisine_coeffs = cuisine_coeffs + intercept
    return cuisine_coeffs

def get_standard_error_sum(results, covariates):
    '''
    #95CI is approximated with +- 2 sum_variance_standard_error
    '''
    # get the variance covariance matrix
    # print(covariates)
    vcov = results.cov_params() \
        .loc[covariates, covariates].values

    # calculate the sum of all pair wise covariances by summing up off-diagonal entries
    off_dia_sum = np.sum(vcov)
    # variance of a sum of variables is the square root
    return np.sqrt(off_dia_sum)

def get_abs_errs(res, ref='us'):
    abs_errs_per_coeff = {}
    for region_coeff in res.params.filter(like='_region',axis=0).index:
        covariates = ['Intercept'] + [region_coeff]
        err = get_standard_error_sum(res, covariates)
        abs_errs_per_coeff[region_coeff.split('T.')[-1].replace(']','')] = err
    abs_errs_per_coeff[ref] = res.bse['Intercept']
    return abs_errs_per_coeff

def get_abs_per_cuisine_errs(res):
    abs_errs_per_coeff = {}
    for cuisine_coeff in res.bse.filter(like='biz_cuisine',axis=0).index:
        covariates = ['const'] + [cuisine_coeff]
        err = get_standard_error_sum(res, covariates)
        abs_errs_per_coeff[cuisine_coeff.split('_')[-1]] = err
    return abs_errs_per_coeff

def get_pvalues(res, ref='us'):
    out = {ref: res.pvalues['Intercept']}
    for region_coeff in res.params.filter(like='_region',axis=0).index:
        out[region_coeff.split('T.')[-1].replace(']','')] = res.pvalues[region_coeff]
    return out

def get_per_cuisine_pvals(res):
    cuisine_coeffs = res.pvalues.filter(like='biz_cuisine',axis=0)
    cuisine_coeffs.index = [x.replace('biz_cuisine_','') for x in cuisine_coeffs.index]
    return cuisine_coeffs

def save_res(res, savename):
    df = pd.DataFrame(res.summary().tables[1])
    df.columns=df.iloc[0]
    df = df[1:]
    df.to_csv(savename, index=False)
    
def _do_regression(df, dep_var, cuisine_ind_var, cuisine_ref, covariates, out_dir):
    
    if cuisine_ind_var != 'biz_cuisine':
        formula = f"{dep_var} ~ C({cuisine_ind_var}, Treatment(reference='{cuisine_ref}'))"
        if 'review_len' in covariates:
            formula += ' + review_len'
        if 'biz_price_point' in covariates:
            formula += ' + C(biz_price_point, Treatment(reference=2))'
        if 'biz_mean_star_rating' in covariates:
            formula += ' + biz_mean_star_rating'
        if 'biz_median_nb_income' in covariates:
            formula += ' + biz_median_nb_income'
        if 'biz_nb_diversity' in covariates:
            formula += ' + biz_nb_diversity'
        print(f"\nPerforming regression with the following formula: {formula}")

        mod = smf.ols(formula=formula, data=df)
    else:
        Y = df[dep_var]
        X = df[[x for x in covariates if x != 'biz_price_point']]
        X = sm.add_constant(X)
        biz_price_point = pd.get_dummies(df['biz_price_point'], prefix='biz_price_point', drop_first=False)
        biz_price_point.drop('biz_price_point_2', axis = 1, inplace=True)
        biz_cuisine = pd.get_dummies(df['biz_cuisines'].explode()).groupby(level=0).sum()
        biz_cuisine.columns = [f'biz_cuisine_{x}' for x in biz_cuisine]
        fullX = pd.concat([X, biz_price_point, biz_cuisine], axis=1)

        mod = sm.OLS(Y, fullX)
    
    modf = mod.fit()
    print()
    print(modf.summary())
    
    if cuisine_ind_var != 'biz_cuisine':
        abs_coeffs = get_abs_coeffs(modf, ref='us')
        abs_errs = get_abs_errs(modf, ref='us')
        pvalues = get_pvalues(modf, ref='us')
    else:
        abs_coeffs = get_abs_per_cuisine_coeffs(modf)
        abs_errs = get_abs_per_cuisine_errs(modf)
        pvalues = get_per_cuisine_pvals(modf)
    print("\tabsolute coeffs:",abs_coeffs)
    print("\tabsolute errs:",abs_errs)
    print("\tpvalues:",pvalues)
    
    savefile = os.path.join(out_dir, f"{dep_var}_{cuisine_ind_var}.csv")
    save_res(modf, savefile)
    savefile = os.path.join(out_dir, f"{dep_var}_{cuisine_ind_var}_covars.csv")
    modf.cov_params().to_csv(savefile)

def do_all_regressions(out_dir, df):
    
    print("\nDoing Study 1 regressions on full reviews set...")
    covariates = ['review_len','biz_price_point','biz_mean_star_rating','biz_median_nb_income','biz_nb_diversity']
    for dep_var in ['exotic_words_agg_score','auth_words_agg_score','typic_words_agg_score']:
        for cuisine_ind_var in ['biz_macro_region','biz_cuisine_region','biz_cuisine']:
            _do_regression(df, dep_var, cuisine_ind_var, 'us', covariates, out_dir)
            
    # add race-othering
    
#     # Study 2
#     print("\nDoing Study 2 regressions on full reviews set...")
#     covariates = ['review_len','biz_price_point','biz_mean_star_rating','biz_median_nb_income','biz_nb_diversity']
#     for dep_var in ['filtered_liwc_posemo_agg_score','luxury_words_agg_score',
#                     'hygiene_words_agg_score','hygiene_pos_words_agg_score','hygiene_neg_words_agg_score',
#                     'cheapness_words_agg_score','cheapness_exp_words_agg_score','cheapness_cheap_words_agg_score']:
#         for cuisine_ind_var in ['biz_macro_region','biz_cuisine_region','biz_cuisine']:
#             _do_regression(df, dep_var, cuisine_ind_var, 'europe', covariates, out_dir)
    # Study 2 glass ceiling
    
    # top-cuisine removed
    # Study 1 cajun-creole removed
    # user-controlled
    
    
def main(path_to_reviews_df, out_dir, text_fields, batch_size, start_batch_no, end_batch_no, debug):
    reviews = load_reviews_df(path_to_reviews_df, debug)
    reviews = zscore_df(reviews)
    if not debug:
        check_VIF(reviews)
    else:
        print("Debug mode ON; skipping VIF step")
    do_all_regressions(out_dir, reviews)
    
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_reviews_df', type=str, default='../data/yelp/restaurants_only/per_reviews_df.csv',
                        help='where to read in reviews dataframe from')
    parser.add_argument('--out_dir', type=str, default='results/',
                        help='directory to save output to')
    parser.add_argument('--debug', action='store_true',
                        help='whether to run on subset of data for debugging purposes')
    args = parser.parse_args()
    if not args.debug:
        print("\n******WARNING****** DEBUG MODE OFF!")
    else:
        print("\nRunning in debug mode; will skip VIF scores check.")
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    main(args.path_to_reviews_df, args.out_dir, args.debug)
    