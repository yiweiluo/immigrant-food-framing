#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf

TOP_CUISINES = set(['american (traditional)','american (new)','cajun/creole','southern','soul food',
                    'mexican','latin american','cuban',
                    'italian','mediterranean','greek','french','irish','spanish',
                    'chinese','japanese','thai','vietnamese','indian','korean',])

def load_restaurants_df(path_to_restaurants_df):
    restaurant_data = pd.read_csv(path_to_restaurants_df,index_col=0)
    print(f"\nRead in census enriched restaurant data from {path_to_restaurants_df} with {len(restaurant_data)} rows.")

    # Annotate for racial demographics
    restaurant_data['total_pop'] = restaurant_data['Population of one race'] + restaurant_data['Population of two or more races']
    restaurant_data['pct_asian'] = restaurant_data['Population of one race: Asian alone'] / restaurant_data['total_pop']
    restaurant_data['pct_hisp'] = restaurant_data['Percentage hispanic']
    return restaurant_data

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
    
    print("\nHydrating reviews df with user data...")
    review_id2user_id = pd.read_csv('../data/yelp/review_id2user_id.csv')
    review_id2user_id = dict(zip(review_id2user_id['review_id'], review_id2user_id['user_id']))
    reviews_df['user_id'] = reviews_df['review_id'].apply(lambda x: review_id2user_id[x])
    print("\tDone!")
    
    return reviews_df

def zscore_df(df, anchor='agg'):
    print(f"\nz-scoring non-dummy variables...")
    
    for feat in ["review_len","biz_mean_star_rating","biz_median_nb_income","biz_nb_diversity",'biz_nb_pct_asian','biz_nb_pct_hisp']:
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

def save_res(res, savename, user_controlled=False):
    df = pd.DataFrame(res.summary().tables[1])
    if user_controlled:
        df.to_csv(savename, index=True)
    else:
        df.columns=df.iloc[0]
        df = df[1:]
        df.to_csv(savename, index=False)
    
def _do_regression(df, dep_var, cuisine_ind_var, cuisine_ref, price_ref, covariates, out_dir, prefix, user_controlled=False, overwrite=False):
    
    savefile = os.path.join(out_dir, f"{prefix}{dep_var}_{cuisine_ind_var}.csv")
    if not overwrite and os.path.exists(savefile):
        print(f"\nRegression of {dep_var} on {cuisine_ind_var} with {prefix} already done; skipping...")
    else:
        if cuisine_ind_var != 'biz_cuisine':
            formula = f"{dep_var} ~ C({cuisine_ind_var}, Treatment(reference='{cuisine_ref}'))"
            if 'review_len' in covariates:
                formula += ' + review_len'
            if 'biz_price_point' in covariates:
                formula += f' + C(biz_price_point, Treatment(reference={price_ref}))'
            if 'biz_mean_star_rating' in covariates:
                formula += ' + biz_mean_star_rating'
            if 'biz_median_nb_income' in covariates:
                formula += ' + biz_median_nb_income'
            if 'biz_nb_diversity' in covariates:
                formula += ' + biz_nb_diversity'
            print(f"\nPerforming regression with the following formula: {formula}")
            
            if user_controlled:
                mod = smf.mixedlm(formula=formula, data=df, groups=df['user_id'])
            else:
                mod = smf.ols(formula=formula, data=df)
        else:
            Y = df[dep_var]
            X = df[[x for x in covariates if x != 'biz_price_point']]
            X = sm.add_constant(X)
            biz_price_point = pd.get_dummies(df['biz_price_point'], prefix='biz_price_point', drop_first=False)
            biz_price_point.drop(f'biz_price_point_{price_ref}', axis = 1, inplace=True)
            biz_cuisine = pd.get_dummies(df['biz_cuisines'].explode()).groupby(level=0).sum()
            biz_cuisine.columns = [f'biz_cuisine_{x}' for x in biz_cuisine]
            fullX = pd.concat([X, biz_price_point, biz_cuisine], axis=1)

            if user_controlled:
                mod = sm.MixedLM(Y, fullX, df['user_id'])
            else:
                mod = sm.OLS(Y, fullX)

        modf = mod.fit()
        print()
        print(modf.summary())

        if cuisine_ind_var != 'biz_cuisine':
            abs_coeffs = get_abs_coeffs(modf, ref=cuisine_ref)
            abs_errs = get_abs_errs(modf, ref=cuisine_ref)
            pvalues = get_pvalues(modf, ref=cuisine_ref)
        else:
            abs_coeffs = get_abs_per_cuisine_coeffs(modf)
            abs_errs = get_abs_per_cuisine_errs(modf)
            pvalues = get_per_cuisine_pvals(modf)
        print("\tabsolute coeffs:",abs_coeffs)
        print("\tabsolute errs:",abs_errs)
        print("\tpvalues:",pvalues)

        save_res(modf, savefile, user_controlled=user_controlled)
        savefile = os.path.join(out_dir, f"{prefix}{dep_var}_{cuisine_ind_var}_covars.csv")
        modf.cov_params().to_csv(savefile)
        
def _do_race_regression(df, restaurants_df, price_ref, out_dir, prefix, user_controlled=False, overwrite=False):
    print("\nDoing Study 1 race effect regressions... first excluding reviews from known visitors...")
    old_len = len(df)
    known_origin_users = set(pickle.load(open('../data/yelp/known_visitors.pkl','rb')))
    df = df.loc[~df['user_id'].isin(known_origin_users)]
    print(f"\tDone! Excluded {old_len - len(df)} reviews.")
    
    print("\nGetting median/IQR pct. race of restaurant neighborhoods...")
    print(restaurants_df[['pct_asian','pct_hisp']].describe())
    hi_asian, lo_asian = df.loc[df['biz_nb_pct_asian']>=restaurants_df['pct_asian'].quantile(.5)], df.loc[df['biz_nb_pct_asian']<=restaurants_df['pct_asian'].quantile(.5)]
    hi_hisp, lo_hisp = df.loc[df['biz_nb_pct_hisp']>=restaurants_df['pct_hisp'].quantile(.5)], df.loc[df['biz_nb_pct_hisp']<=restaurants_df['pct_hisp'].quantile(.5)]
    print("\tNum reviews from hi, lo %Asian neighborhoods:", len(hi_asian), len(lo_asian))
    print("\tNum reviews from hi, lo %Hisp. neighborhoods:", len(hi_hisp), len(lo_hisp))
    
    asian_othering_df = defaultdict(list)
    for dep_var in ['exotic_words_agg_score','auth_words_agg_score','typic_words_agg_score']:
        if user_controlled:
            hi_mod = smf.mixedlm(formula=f"{dep_var} ~ C(biz_price_point, Treatment(reference={price_ref})) + C(biz_cuisine_region, Treatment(reference='us')) + review_len + biz_mean_star_rating + biz_median_nb_income + biz_nb_diversity", data=hi_asian, groups=hi_asian['user_id'])
        else:
            hi_mod = smf.ols(formula=f"{dep_var} ~ C(biz_price_point, Treatment(reference={price_ref})) + C(biz_cuisine_region, Treatment(reference='us')) + review_len + biz_mean_star_rating + biz_median_nb_income + biz_nb_diversity", data=hi_asian)
        hi_modf = hi_mod.fit()
        print(hi_modf.summary())
        asian_othering_df['Pct. Asian'].append('hi')
        if 'exotic' in dep_var:
            asian_othering_df['frame'].append('exoticism')
        if 'auth' in dep_var:
            asian_othering_df['frame'].append('authenticity')
        if 'typic' in dep_var:
            asian_othering_df['frame'].append('typicality')
        asian_othering_df['Othering score'].append(hi_modf.params["C(biz_cuisine_region, Treatment(reference='us'))[T.asia]"])
        asian_othering_df['Othering err'].append(hi_modf.bse["C(biz_cuisine_region, Treatment(reference='us'))[T.asia]"])

        if user_controlled:
            lo_mod = smf.mixedlm(formula=f"{dep_var} ~ C(biz_price_point, Treatment(reference={price_ref})) + C(biz_cuisine_region, Treatment(reference='us')) + review_len + biz_mean_star_rating + biz_median_nb_income + biz_nb_diversity", data=lo_asian, groups=lo_asian['user_id'])
        else:
            lo_mod = smf.ols(formula=f"{dep_var} ~ C(biz_price_point, Treatment(reference={price_ref})) + C(biz_cuisine_region, Treatment(reference='us')) + review_len + biz_mean_star_rating + biz_median_nb_income + biz_nb_diversity", data=lo_asian)
        lo_modf = lo_mod.fit()
        print(lo_modf.summary())
        asian_othering_df['Pct. Asian'].append('lo')
        if 'exotic' in dep_var:
            asian_othering_df['frame'].append('exoticism')
        if 'auth' in dep_var:
            asian_othering_df['frame'].append('authenticity')
        if 'typic' in dep_var:
            asian_othering_df['frame'].append('typicality')
        asian_othering_df['Othering score'].append(lo_modf.params["C(biz_cuisine_region, Treatment(reference='us'))[T.asia]"])
        asian_othering_df['Othering err'].append(lo_modf.bse["C(biz_cuisine_region, Treatment(reference='us'))[T.asia]"])
    asian_othering_df = pd.DataFrame(asian_othering_df)
    print(asian_othering_df)
    asian_othering_df.to_csv(os.path.join(out_dir, f'{prefix}asian_othering_median.csv'))

    hisp_othering_df = defaultdict(list)
    for dep_var in ['exotic_words_agg_score','auth_words_agg_score','typic_words_agg_score']:
        if user_controlled:
            hi_mod = smf.mixedlm(formula=f"{dep_var} ~ C(biz_price_point, Treatment(reference={price_ref})) + C(biz_cuisine_region, Treatment(reference='us')) + review_len + biz_mean_star_rating + biz_median_nb_income + biz_nb_diversity", data=hi_hisp, groups=hi_hisp['user_id'])
        else:
            hi_mod = smf.ols(formula=f"{dep_var} ~ C(biz_price_point, Treatment(reference={price_ref})) + C(biz_cuisine_region, Treatment(reference='us')) + review_len + biz_mean_star_rating + biz_median_nb_income + biz_nb_diversity", data=hi_hisp)
        hi_modf = hi_mod.fit()
        print(hi_modf.summary())
        hisp_othering_df['Pct. Hisp.'].append('hi')
        if 'exotic' in dep_var:
            hisp_othering_df['frame'].append('exoticism')
        if 'auth' in dep_var:
            hisp_othering_df['frame'].append('authenticity')
        if 'typic' in dep_var:
            hisp_othering_df['frame'].append('typicality')
        hisp_othering_df['Othering score'].append(hi_modf.params["C(biz_cuisine_region, Treatment(reference='us'))[T.latin_america]"])
        hisp_othering_df['Othering err'].append(hi_modf.bse["C(biz_cuisine_region, Treatment(reference='us'))[T.latin_america]"])

        if user_controlled:
            lo_mod = smf.mixedlm(formula=f"{dep_var} ~ C(biz_price_point, Treatment(reference={price_ref})) + C(biz_cuisine_region, Treatment(reference='us')) + review_len + biz_mean_star_rating + biz_median_nb_income + biz_nb_diversity", data=lo_hisp, groups=lo_hisp['user_id'])
        else:
            lo_mod = smf.ols(formula=f"{dep_var} ~ C(biz_price_point, Treatment(reference={price_ref})) + C(biz_cuisine_region, Treatment(reference='us')) + review_len + biz_mean_star_rating + biz_median_nb_income + biz_nb_diversity", data=lo_hisp)
        lo_modf = lo_mod.fit()
        print(lo_modf.summary())
        hisp_othering_df['Pct. Hisp.'].append('lo')
        if 'exotic' in dep_var:
            hisp_othering_df['frame'].append('exoticism')
        if 'auth' in dep_var:
            hisp_othering_df['frame'].append('authenticity')
        if 'typic' in dep_var:
            hisp_othering_df['frame'].append('typicality')
        hisp_othering_df['Othering score'].append(lo_modf.params["C(biz_cuisine_region, Treatment(reference='us'))[T.latin_america]"])
        hisp_othering_df['Othering err'].append(lo_modf.bse["C(biz_cuisine_region, Treatment(reference='us'))[T.latin_america]"])
    hisp_othering_df = pd.DataFrame(hisp_othering_df)
    print(hisp_othering_df)
    hisp_othering_df.to_csv(os.path.join(out_dir, f'{prefix}hisp_othering_median.csv'))

def do_all_regressions(out_dir, prefix, df, restaurants_df, cuisines_to_remove=set(), user_controlled=False):
    
    if len(cuisines_to_remove) > 0:
        print("\nDoing OLS regressions with the following cuisines removed:", cuisines_to_remove)
        df = df.loc[df['biz_cuisines'].apply(lambda x: len(set(x).intersection(cuisines_to_remove))) == 0]
        print("\tNew cuisine distribution:")
        for cuisine in TOP_CUISINES:
            print(cuisine, len(df.loc[df['biz_cuisines'].apply(lambda x: cuisine in x)]))
        print("\n\tNew cuisine region distribution:")
        print(df['biz_cuisine_region'].value_counts())
        
    if user_controlled:
        print("\nDoing mixed effect regressions among superyelper reviews (>100 reviews) with random effects per user...")
        top_users = sorted(Counter(df['user_id']).items(), key=lambda x: x[1], reverse=True)
        review_thresh = 100
        top_user_ids = set([x[0] for x in top_users if x[1] >= review_thresh])
        print(f"\tNum. high-volume users contributing {review_thresh} or more reviews:", len(top_user_ids))
        df = df.loc[df['user_id'].isin(top_user_ids)]
        print(f"\tNum reviews by these users: {len(df)}")
        print("\nChecking skew of users over cuisines they review...")
        xtab = pd.crosstab(df['user_id'], df['biz_cuisine_region'], normalize='index')
        print(xtab)
        even_users = xtab.loc[(xtab['asia'] >= 0.1) & 
                              (xtab['latin_america'] >= 0.1) & 
                              (xtab['europe'] >= 0.1)].index.values
        print("\tNum unskewed users with at least 10% of reviews in each non-US cuisine:",len(even_users))
        df = df.loc[df['user_id'].isin(even_users)]
        print("\tNum unskewed superyelper reviews:", len(df))
        print("\nDistribution over cuisines of unskewed superyelper reviews:")
        print(df['biz_cuisine_region'].value_counts())
    
#     # Study 1
#     print("\nDoing Study 1 regressions...")
#     covariates = ['review_len','biz_price_point','biz_mean_star_rating','biz_median_nb_income','biz_nb_diversity']
#     for dep_var in ['exotic_words_agg_score','auth_words_agg_score','auth_simple_words_agg_score','auth_other_words_agg_score','typic_words_agg_score']:
#         for cuisine_ind_var in ['biz_macro_region','biz_cuisine_region','biz_cuisine']:
#             _do_regression(df, dep_var, cuisine_ind_var, 'us', 2, covariates, out_dir, prefix, user_controlled=user_controlled, overwrite=True)
            
#     # Study 1 race-othering
#     _do_race_regression(df, restaurants_df, 2, out_dir, prefix, user_controlled=user_controlled, overwrite=False)
    
#     # Study 2
#     print("\nDoing Study 2 regressions...")
#     covariates = ['review_len','biz_price_point','biz_mean_star_rating','biz_median_nb_income','biz_nb_diversity']
#     for dep_var in ['filtered_liwc_posemo_agg_score','luxury_words_agg_score',
#                     'hygiene_words_agg_score','hygiene_pos_words_agg_score','hygiene_neg_words_agg_score',
#                     'cheapness_words_agg_score','cheapness_exp_words_agg_score','cheapness_cheap_words_agg_score']:
#         for cuisine_ind_var in ['biz_cuisine_region','biz_cuisine']:
#             _do_regression(df, dep_var, cuisine_ind_var, 'europe', 2, covariates, out_dir, prefix, user_controlled=user_controlled, overwrite=True)
    
    # Study 2 glass ceiling
    print("\nDoing Study 2 regressions within $$$-$$$$ price point restaurants...")
    covariates = ['review_len','biz_price_point','biz_mean_star_rating','biz_median_nb_income','biz_nb_diversity']
    for dep_var in ['filtered_liwc_posemo_agg_score','luxury_words_agg_score',
                    'hygiene_words_agg_score','hygiene_pos_words_agg_score','hygiene_neg_words_agg_score',
                    'cheapness_words_agg_score','cheapness_exp_words_agg_score','cheapness_cheap_words_agg_score']:
        for cuisine_ind_var in ['biz_cuisine_region','biz_cuisine']:
            _do_regression(df.loc[df['biz_price_point'].isin({3,4})], dep_var, cuisine_ind_var, 'europe', 3, covariates, out_dir, 
                           f'{prefix}glass_ceiling_', user_controlled=user_controlled, overwrite=True)

def main(path_to_restaurants_df, path_to_reviews_df, out_dir, debug):
    restaurants = load_restaurants_df(path_to_restaurants_df)
    reviews = load_reviews_df(path_to_reviews_df, debug)
    reviews = zscore_df(reviews)
    if not debug:
        check_VIF(reviews)
    else:
        print("Debug mode ON; skipping VIF step")
    do_all_regressions(out_dir, '', reviews, restaurants)
    do_all_regressions(out_dir, 'top_removed_', reviews, restaurants, {'american (traditional)', 'italian', 'mexican', 'chinese'})
    do_all_regressions(out_dir, 'cajun-creole_removed_', reviews, restaurants, {'cajun/creole'})
    do_all_regressions(out_dir, 'user_cont_', reviews, restaurants, user_controlled=True)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_restaurants_df', type=str, default='../data/yelp/census_enriched_business_data.csv',
                        help='where to read in restaurants dataframe from')
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
        
    main(args.path_to_restaurants_df, args.path_to_reviews_df, args.out_dir, args.debug)
    