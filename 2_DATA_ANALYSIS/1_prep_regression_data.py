#!/usr/bin/env python
# coding: utf-8

import os, glob, json
import pickle
import pandas as pd
import argparse
from tqdm import tqdm, trange
from collections import Counter, defaultdict


CHAIN_THRESHOLD = 1
CATEGORIES_TO_EXCLUDE = {'cafe','fast food'}
REGIONS = {'us','europe','latin_america','asia'}
TOP_CUISINES = set(['american (traditional)','american (new)','italian','mexican','chinese','japanese',
                    'asian fusion','mediterranean','thai','cajun/creole','latin american','southern',
                    'vietnamese','indian','greek','caribbean','middle eastern','french','soul food',
                    'korean','tex-mex','cuban','spanish','irish'])

def _is_restaurant(cat_set, search_set={'restaurants','food'}):
    return len(cat_set.intersection(search_set)) > 0

def _clean_list(l):
    return [x.lower().strip() for x in l if x is not None]

def _is_homogeneous(continents, verbose=False):
    if verbose:
        print(set(continents))
    if 'fusion' in continents:
        return 'fusion'
    return 'homo' if len(set(continents)) == 1 else 'hetero'

def prep_census_enriched_df(path_to_enriched_df):
    """Read in business data with census data-enriched demographic fields; then annotate with additional fields and subset to restaurants."""
    
    restaurant_data = pd.read_csv(path_to_enriched_df,index_col=0)
    print(f"\nRead in census enriched business data from {path_to_enriched_df} with {len(restaurant_data)} rows.")

    # Filter to restaurant data
    restaurant_data['categories'] = restaurant_data['categories'].apply(lambda x: x.split(',') if type(x) == str else [])
    restaurant_data['categories'] = restaurant_data['categories'].apply(lambda x: set(_clean_list(x)))
    restaurant_data = restaurant_data.loc[restaurant_data['categories'].apply(lambda x: _is_restaurant(x) == True)].copy()
    print(f"\nFiltered to restaurant data with {len(restaurant_data)} rows.")

    # Annotate for racial demographics
    restaurant_data['total_pop'] = restaurant_data['Population of one race'] + restaurant_data['Population of two or more races']
    restaurant_data['pct_asian'] = restaurant_data['Population of one race: Asian alone'] / restaurant_data['total_pop']

    # Annotate for continent
    ethnic_cats_per_continent = pd.read_csv('../ethnic_cats_per_continent.csv')
    ethnic_cats_per_continent = ethnic_cats_per_continent.loc[ethnic_cats_per_continent['region'].isin(REGIONS)]
    ethnic_cat2continent = dict(zip(ethnic_cats_per_continent['cuisine'],ethnic_cats_per_continent['region']))
    print(f"\nAnnotating for the following geographic regions: {REGIONS}")
    restaurant_data['continents'] = restaurant_data['categories'].apply(
        lambda x: [ethnic_cat2continent[cat] for cat in x if cat in ethnic_cat2continent])
    restaurant_data['is_homogeneous_or_fusion'] = restaurant_data['continents'].apply(lambda x: _is_homogeneous(x))
    print("\nDistribution of regionally homogeneous or fusion restaurants:")
    print(restaurant_data['is_homogeneous_or_fusion'].value_counts())

    # Annotate for price level
    print("\nAnnotating for price level...")
    biz2price = {}
    for _,row in tqdm(restaurant_data.iterrows()):
        biz_id = row['business_id']
        if type(row['attributes']) == str:
            attrs = json.loads(row['attributes'].replace("\'",'"').replace('"u"','"').replace('u"','"').replace('""','"')\
                                                .replace(': "{',': {').replace('}",','},').replace('}"}','}}')\
                                                .replace('False','false').replace('True','true').replace('None','false'))
            try:
                price = attrs['RestaurantsPriceRange2']
            except KeyError:
                try:
                    price = attrs['RestaurantsPriceRange1']
                except KeyError:
                    price = None
        else:
            price = None
        biz2price[biz_id] = price
    restaurant_data['price_level'] = restaurant_data['business_id'].apply(lambda x: biz2price[x])
    print(restaurant_data['price_level'].value_counts(normalize=True).sort_index())

    # Annotate for whether a restaurant is a chain
    print(f"\nAnnotating for chains using threshold of >{CHAIN_THRESHOLD}...")
    restaurant_counts = restaurant_data['name'].value_counts()
    chain_names = set([r for r in restaurant_counts.index if restaurant_counts[r] > CHAIN_THRESHOLD])
    chain_ids = set(restaurant_data.loc[restaurant_data['name'].isin(set(chain_names))]['business_id'].values)
    restaurant_data['is_chain'] = restaurant_data['business_id'].apply(lambda x: x in chain_ids)
    print(restaurant_data['is_chain'].value_counts())
    print()
    print(restaurant_data.head())
    
    return restaurant_data

def filter_businesses_for_regression(restaurant_data):
    """Filter restaurant data"""
    
    print(f"\nFiltering from {len(restaurant_data)} total restaurants for regression...")
    
    print("\tExcluding chains...")
    dat = restaurant_data.loc[restaurant_data['is_chain']==False].copy()
    print(f"\t\tNew #restaurants: {len(dat)}")
    
    print(f"\tExcluding following categories: {CATEGORIES_TO_EXCLUDE}...")
    dat = dat.loc[dat['categories'].apply(lambda x: len(x.intersection(CATEGORIES_TO_EXCLUDE))==0)].copy()
    print(f"\t\tNew #restaurants: {len(dat)}")
    
    print("\tExcluding multi-region restaurants...")
    dat = dat.loc[dat['is_homogeneous_or_fusion'].apply(lambda x: x == 'homo' or x == 'fusion')].copy()
    print(f"\t\tNew #restaurants: {len(dat)}")
    
    print(f"\tSubsetting to top 20 cuisines: {TOP_CUISINES}...")
    dat = dat.loc[dat['categories'].apply(lambda x: len(set(x).intersection(TOP_CUISINES)) > 0)].copy()
    print(f"\t\tNew #restaurants: {len(dat)}")
    
    print(f"\tExcluding restaurants without price point label; converting price points to integers...")
    dat = dat.loc[(dat['price_level']!='false') & 
                  (~pd.isna(dat['price_level']))].copy()
    dat['price_level'] = dat['price_level'].apply(lambda x: int(x))
    print(f"\t\tDone! New #restaurants: {len(dat)}")
    
    print("\nCuisine distribution in filtered restaurant data:")
    for cuisine in TOP_CUISINES:
        print(cuisine, len(dat.loc[dat['categories'].apply(lambda x: cuisine in x)]))

    return dat

def load_raw_reviews(path_to_raw_reviews):
    """Load raw reviews to get review-business relationships and review lengths."""
    
    print("\nLoading in raw reviews...")
    file_sep = ',' if path_to_raw_reviews.endswith('.csv') else '\t'
    raw_df = pd.read_csv(path_to_raw_reviews, sep=file_sep)
    print(f"\tRead in {len(raw_df)} reviews.")
    print("\nGetting review lengths...")
    raw_df['len'] = raw_df['text'].apply(lambda x: len(x.split()))
    print("\tDone! Review length distribution:")
    print(raw_df['len'].describe())
    
    return raw_df

def get_filtered_business_reviews(filtered_restaurant_data, raw_reviews):
    """Get IDs of reviews associated with filtered restaurants"""
    
    print(f"\nGetting review IDs associated with {len(filtered_restaurant_data)} filtered restaurants...")
    review_ids = raw_reviews.loc[raw_reviews['review_id'].isin(set(filtered_restaurant_data['business_id'].values))]['review_id'].values
    print(f"\tDone! Found {len(review_ids)} reviews.")
    print("Sample review IDs:", review_ids[:5])

    return review_ids

def _hydrate_with_biz_data():

def create_reviews_df(review_ids_for_df, raw_reviews):
    review_id2len = dict(zip(raw_reviews['review_id'], raw_reviews['len']))
    
    per_review_df = defaultdict(list)                        
 
    for review_id in tqdm(review_ids_for_df):
        per_review_df['review_id'].append(review_id)
        per_review_df['review_len'].append()
        review_frames = frames_lookup[review_id]
        for feat in set(avail_feats).difference(to_ignore):
            full_agg_score = 0
            full_agg_matches = Counter()
            no_neg_agg_score = 0
            no_neg_agg_matches = Counter()
            for anchor_type in ['food','service','establishment']:
                anchor_frames = [x for x in review_frames
                                 if x[2].replace('_',' ') in anchor_type2anchors[anchor_type]
                                 or x[3].replace('_',' ') in anchor_type2anchors[anchor_type]]
    #             anchor_min_bf_frames = [x for x in anchor_frames 
    #                                     if (x[1] in frame_freqs_lookup)
    #                                     and (frame_freqs_lookup[x[1]][anchor_type]['biz'] >= min_biz_freq)]
                anchor_frames_with_neg = [x[1] for x in anchor_frames]
                anchor_frames_no_neg = [x[1] for x in anchor_frames
                                        if len(set(x[0].split(',')).intersection(NEGATIONS)) == 0]
                full_res = score_dict_feat(anchor_frames_with_neg, feat=feat)
                full_score = full_res[0]
                full_matches = full_res[1]
    #                 per_review_df[f"{anchor_type}_{feat}_score"].append(score)
    #                 per_review_df[f"{anchor_type}_{feat}_matches"].append(matches)
                full_agg_score += full_score
                full_agg_matches += full_matches
                no_neg_res = score_dict_feat(anchor_frames_no_neg, feat=feat)
                no_neg_score = no_neg_res[0]
                no_neg_matches = no_neg_res[1]
    #                 per_review_df[f"{anchor_type}_{feat}_score"].append(score)
    #                 per_review_df[f"{anchor_type}_{feat}_matches"].append(matches)
                no_neg_agg_score += no_neg_score
                no_neg_agg_matches += no_neg_matches
            per_review_df[f"agg_{feat}_score"].append(full_agg_score)
            per_review_df[f"agg_{feat}_matches"].append(full_agg_matches)
            per_review_df[f"no_neg_agg_{feat}_score"].append(no_neg_agg_score)
            per_review_df[f"no_neg_agg_{feat}_matches"].append(no_neg_agg_matches)

    #         per_review_df['biz_id'].append(biz_key)
    #         per_review_df['biz_mean_star_rating'].append(stars)
    #         per_review_df['biz_price_level'].append(price_level)
    #         per_review_df['biz_median_nb_income'].append(nb_income)
    #         per_review_df['biz_nb_diversity'].append(nb_diversity)
    #         per_review_df['biz_cuisines'].append(cuisines)
    #         per_review_df['biz_region'].append(region)
    #     i += 1
    #     if i > 20:
    #         break

    per_review_df = pd.DataFrame(per_review_df)             
    display(per_review_df.head())                          
    print(len(per_review_df))  
    per_review_df.to_pickle('per_review_df.pkl')
    
    return reviews_df

    
def main(path_to_enriched_df, path_to_raw_reviews, out_dir, text_fields, batch_size, start_batch_no, end_batch_no, debug):
    restaurants = prep_census_enriched_df(path_to_enriched_df)
    filtered_restaurants = filter_businesses_for_regression(restaurants)
    raw_reviews = load_raw_reviews(path_to_raw_reviews)
    review_ids = get_filtered_business_reviews(filtered_restaurants, raw_reviews)
    #create_reviews_df(filtered_restaurants, raw_reviews)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_enriched_df', type=str, default='../data/yelp/census_enriched_business_data.csv',
                        help='where to read in census enriched dataframe from')
    parser.add_argument('--path_to_raw_reviews', type=str, default='../data/yelp/restaurants_only/restaurant_reviews_df.csv',
                        help='where to read in raw reviews dataframe from')
    parser.add_argument('--out_dir', type=str, default='../data/yelp/restaurants_only/spacy_processed',
                        help='directory to save output to')
    parser.add_argument('--text_fields', type=str, default='text',
                        help='column name(s) for text fields')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='batch size for spaCy')
    parser.add_argument('--start_batch_no', type=int, default=0, 
                        help='batch number to start at')
    parser.add_argument('--end_batch_no', type=int, default=2000,
                        help='batch number to end before (non-inclusive)')
    parser.add_argument('--debug', action='store_true',
                        help='whether to run on subset of data for debugging purposes')
    args = parser.parse_args()
    if not args.debug:
        print("\n******WARNING****** DEBUG MODE OFF!")
    else:
        print("\nRunning in debug mode; will limit to processing first batch of texts with batch size of 10.")
        args.start_batch_no, args.end_batch_no, args.batch_size = 0, 1, 10
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    main(args.path_to_enriched_df, args.path_to_raw_reviews, args.out_dir, args.text_fields, args.batch_size, args.start_batch_no, args.end_batch_no, args.debug)
    