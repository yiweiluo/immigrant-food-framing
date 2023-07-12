#!/usr/bin/env python
# coding: utf-8

import os, glob
import json
import pickle
import pandas as pd
import argparse
from tqdm import tqdm
from collections import Counter, defaultdict
import time
import math

with open('../anchor_sets/food_anchors.txt','r') as f:
    food_anchors = set(f.read().splitlines())
with open('../anchor_sets/establishment_anchors.txt','r') as f:
    establishment_anchors = set(f.read().splitlines())
with open('../anchor_sets/waitstaff_anchors.txt','r') as f:
    service_anchors = set(f.read().splitlines())
print(len(food_anchors), len(establishment_anchors), len(service_anchors))
anchor_type2anchors = {
    'food': food_anchors, 'service': service_anchors, 'establishment': establishment_anchors}
all_anchors = set(anchor_type2anchors['food']) | set(anchor_type2anchors['service']) | \
                set(anchor_type2anchors['establishment'])

def load_reviews_df(path_to_reviews_df, do_yelp=True):
    print("\nReading in reviews_df...")
    reviews_df = pd.read_pickle(path_to_reviews_df)
    reviews_df['biz_macro_region'] = reviews_df['biz_cuisine_region'].apply(lambda x: 'us' if x == 'us' else 'non-us')
    reviews_df['biz_cuisines'] = reviews_df['biz_cuisines'].apply(lambda x: list(x))

    print(f"\tDone! Read in df with shape {reviews_df.shape}.")
    print(reviews_df.head())
    print()
    print(reviews_df['biz_macro_region'].value_counts())
    print()
    print(reviews_df['biz_cuisine_region'].value_counts())
    if do_yelp:
        print()
        print(reviews_df[['biz_median_nb_income','biz_nb_diversity']].describe())
    
        print("\nHydrating reviews df with user data...")
        review_id2user_id = pd.read_csv('../data/yelp/review_id2user_id.csv')
        review_id2user_id = dict(zip(review_id2user_id['review_id'], review_id2user_id['user_id']))
        reviews_df['user_id'] = reviews_df['review_id'].apply(lambda x: review_id2user_id[x])
        print("\tDone!")
    else:
        print(reviews_df['biz_sentiment'].value_counts())
        print(reviews_df['biz_sentiment'].value_counts(normalize=True))
        
        print("\nSubsampling LLM reviews to be stratified evenly across covariates besides length and sentiment...")
        print(reviews_df['biz_cuisine_region'].value_counts())
        min_cuisine = reviews_df['biz_cuisine_region'].value_counts().min()
        reviews_df = pd.concat([reviews_df.loc[reviews_df['biz_cuisine_region']==region].sample(n=min_cuisine,replace=False)
           for region in ['asia','europe','us','latin_america']])
        print(reviews_df['biz_cuisine_region'].value_counts())
        print(reviews_df['biz_sentiment'].value_counts(normalize=True))
        print("\tDone!")
        time.sleep(3)
    
        reviews_df['biz_price_point'] = reviews_df['biz_price_point'].apply(lambda x: {'$ ($10 and under)':1, 
                                                                               '$$ ($10-$25)':2,
                                                                               '$$$ ($25-$45)':3,
                                                                               '$$$$ ($50 and up)':4}[x])
    print(reviews_df['biz_price_point'].value_counts())
    
    return reviews_df

def load_lookup(path_to_lookup):
    print("\nLoading feature lookup dict...")
    frames_lookup = pickle.load(open(path_to_lookup,'rb'))
    print(f"\tDone! Loaded dict of len {len(frames_lookup)}. Sample lookup keys: {list(frames_lookup.keys())[:3]}.")
    print(f"\tSample frames: {frames_lookup[list(frames_lookup.keys())[0]]}")
    return frames_lookup

def get_fightin_words(counts_1, counts_2, counts_prior=None, verbose=False, counters=True,    
               filter_foods_from_lor=False, food_filter_set='wordnet_freebase_swe_union.txt',
               filter_trivial_modifiers=True, trivial_modifiers_set='wn_demonyms.txt'):
    
    if not counters: # turn lemmas into counts
        counts_1 = Counter(counts_1)
        counts_2 = Counter(counts_2)
        if counts_prior is not None:
            counts_prior = Counter(counts_prior)
    
    if counts_prior == None:
        print("\nNo prior sample passed, using union of samples 1 and 2 as prior.")
        counts_prior = counts_1 + counts_2
        
    print(f"Calculating LOR using samples with token counts of {sum(counts_1.values())}, {sum(counts_2.values())}, {sum(counts_prior.values())}.")
    print("Vocab sizes of counts1, counts2, prior:", len(counts_1), len(counts_2), len(counts_prior))
    
    to_exclude = set()
    if filter_foods_from_lor:
        if food_filter_set == 'freebase':
            freebase = pd.read_csv('freebase_foods.tsv', sep='\t')
            food_names = set(freebase['name'].apply(lambda x: x.lower().strip()).values)
        elif food_filter_set == 'wordnet':
            with open('wordnet_foods.txt','r') as f:
                food_names = set([x.lower() for x in f.read().splitlines()])
        else:
            with open(food_filter_set,'r') as f:
                food_names = set([x.lower() for x in f.read().splitlines()])
        print(f"\nExcluding {len(food_names)} food terms for more meaningful LOR.")
        print(f"\tSample food terms:", list(food_names)[:30])
        to_exclude |= food_names
    if filter_trivial_modifiers:
        with open(trivial_modifiers_set, 'r') as f:
            trivial_modifiers = set([x.lower() for x in f.read().splitlines()])
        print(f"\nExcluding {len(trivial_modifiers)} trivial adjectives for more meaningful LOR.")
        print(f"\tSample trivial adjs:", list(trivial_modifiers)[:30])
        to_exclude |= trivial_modifiers
        
    if len(to_exclude) > 0:
        counts_1 = Counter({w: counts_1[w] 
                           for w in counts_1 if len(set(w.split('_')).intersection(to_exclude)) == 0})
        counts_2 = Counter({w: counts_2[w] 
                           for w in counts_2 if len(set(w.split('_')).intersection(to_exclude)) == 0})
        counts_prior = Counter({w: counts_prior[w] 
                         for w in counts_prior if len(set(w.split('_')).intersection(to_exclude)) == 0})
        print("\tUpdated vocab sizes of counts1, counts2, prior:", len(counts_1), len(counts_2), len(counts_prior))

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in counts_prior.keys():
        counts_prior[word] = int(counts_prior[word] + 0.5)

    for word in counts_2.keys():
        counts_1[word] = int(counts_1[word] + 0.5)
        if counts_prior[word] == 0:
            counts_prior[word] = 1

    for word in counts_1.keys():
        counts_2[word] = int(counts_2[word] + 0.5)
        if counts_prior[word] == 0:
            counts_prior[word] = 1

    n1  = sum(counts_1.values())
    n2  = sum(counts_2.values())
    nprior = sum(counts_prior.values())

    for word in counts_prior.keys():
        if counts_prior[word] > 0:
            l1 = float(counts_1[word] + counts_prior[word]) / (( n1 + nprior ) - (counts_1[word] + counts_prior[word]))
            l2 = float(counts_2[word] + counts_prior[word]) / (( n2 + nprior ) - (counts_2[word] + counts_prior[word]))
            sigmasquared[word] =  1/(float(counts_1[word]) + float(counts_prior[word])) + 1/(float(counts_2[word]) + float(counts_prior[word]))
            sigma[word] =  math.sqrt(sigmasquared[word])
            delta[word] = ( math.log(l1) - math.log(l2) ) / sigma[word]

    if verbose:
        for word in sorted(delta, key=delta.get):
            print(word)
            print("%.3f" % delta[word])
            
    lors_df = pd.DataFrame(delta, index=[0]).T.sort_values(by=0, ascending=False).reset_index()#.sort_
    lors_df.columns = ['lemma','z-score']
    
    return lors_df

def fightin_words_report(out_dir, df, features_lookup, guid, feature_list, NEGATIONS = {'not','no'}):
    if feature_list is None:
        print("\nGetting general fightin words, unrestricted to particular framing dimensions...")
        
        print("\nGetting features per cuisine...")
        frames_per_cuisine = {}
        for cuisine in tqdm(['us','europe','latin_america','asia']):
            cuisine_df = df.loc[df['biz_cuisine_region']==cuisine]
            cuisine_lemmas = []
            for _,row in cuisine_df.iterrows():
                review_frames = features_lookup[row[guid]]
                anchor_frames = [x for x in review_frames
                                 if x[2].replace('_',' ') in all_anchors
                                 or x[3].replace('_',' ') in all_anchors]
                anchor_frames_no_neg = [x[1] for x in anchor_frames
                                        if len(set(x[0].split(',')).intersection(NEGATIONS)) == 0]
                cuisine_lemmas.extend(anchor_frames_no_neg)
            frames_per_cuisine[cuisine] = Counter(cuisine_lemmas)
        
        for region in ['us','europe','latin_america','asia']:
            savename = os.path.join(out_dir, f'most_{region}_features.csv')
            print(f"\nGetting fighting words for {region}, will save results to {savename}...")
            lor_res = get_fightin_words(frames_per_cuisine[region], 
                                     sum([frames_per_cuisine[other_cuisine] for other_cuisine in ['us','europe','latin_america','asia'] if other_cuisine != region], Counter()), counts_prior=None, verbose=False,     
                                     filter_foods_from_lor=True, food_filter_set='../anchor_sets/food_anchors.txt',
                                     filter_trivial_modifiers=True)
            lor_res.to_csv(savename, index=False)
            
    else:
        print("\nGetting fighting words of the following features:", feature_list)

        print("\nGetting feature counts...")
        feat_counts_per_cuisine = defaultdict(dict)
        for feat in tqdm(feature_list):
            for cuisine in ['us','europe','latin_america','asia']:
                matches = df.loc[df['biz_cuisine_region']==cuisine][f"{feat}_matches"].values
                feat_counts_per_cuisine[feat][cuisine] = sum([Counter(json.loads(x)) for x in matches if x != '-1'], Counter())
        print("\tDone!")

        for feat in feature_list:
            for cuisine in ['us','europe','latin_america','asia']:
                savename = os.path.join(out_dir, f'most_{cuisine}_{feat}.csv')
                print(f"\nGetting fighting words for feature {feat}, will save results to {savename}...")
                lor_res = get_fightin_words(feat_counts_per_cuisine[feat][cuisine], 
                                     sum([feat_counts_per_cuisine[feat][other_cuisine] for other_cuisine in ['us','europe','latin_america','asia'] if other_cuisine != cuisine], Counter()), counts_prior=None, verbose=False,     
                                     filter_foods_from_lor=False, food_filter_set='../anchor_sets/food_anchors.txt',
                                     filter_trivial_modifiers=False)
                lor_res.to_csv(savename, index=False)

    print("\tDone!")
    
def main(path_to_reviews_df, path_to_lookup, guid, out_dir, do_yelp):
    reviews = load_reviews_df(path_to_reviews_df, do_yelp=do_yelp)
    feature_lookup = load_lookup(path_to_lookup)
    fightin_words_report(out_dir, reviews, feature_list={'exotic_words_agg','auth_words_agg','typic_words_agg'})
    fightin_words_report(out_dir, reviews, feature_list={'filtered_liwc_posemo_agg','luxury_words_agg','hygiene_words_agg','cheapness_words_agg'})
    fightin_words_report(out_dir, reviews, feature_lookup, guid, feature_list=None)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_reviews_df', type=str, default='../data/yelp/restaurants_only/per_reviews_df.csv',
                        help='where to read in reviews dataframe from')
    parser.add_argument('--path_to_lookup', type=str, default='../data/yelp/restaurants_only/frames_lookup.pkl',
                        help='where to read in feature lookup dict from')
    parser.add_argument('--guid', type=str, default='review_id',
                        help='col name of review GUID')
    parser.add_argument('--out_dir', type=str, default='fightin_words_results/',
                        help='directory to save output to')
    parser.add_argument('--do_yelp', action='store_true',
                        help='whether to run on Yelp reviews or LLM reviews')
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    main(args.path_to_reviews_df, args.path_to_lookup, args.guid, args.out_dir, args.do_yelp)
    