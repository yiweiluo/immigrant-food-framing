#!/usr/bin/env python
# coding: utf-8

import os, glob, json
import pickle, dill
import pandas as pd
import argparse
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import time

CHAIN_THRESHOLD = 1
CATEGORIES_TO_EXCLUDE = {'cafe','fast food'}
REGIONS = {'us','europe','latin_america','asia'}
TOP_CUISINES = set(['american (traditional)','american (new)','cajun/creole','southern','soul food',
                    'mexican','latin american','cuban',
                    'italian','mediterranean','greek','french','irish','spanish',
                    'chinese','japanese','thai','vietnamese','indian','korean',])

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
    raw_df['len'] = raw_df['text'].apply(lambda x: len(x.split( )) if type(x) == str else 0)
    print("\tDone! Review length distribution:")
    print(raw_df['len'].describe())
    
    return raw_df

def get_filtered_business_reviews(filtered_restaurant_data, raw_reviews):
    """Get IDs of reviews associated with filtered restaurants"""
    
    print(f"\nGetting review IDs associated with {len(filtered_restaurant_data)} filtered restaurants...")
    review_ids = raw_reviews.loc[raw_reviews['business_id'].isin(set(filtered_restaurant_data['business_id'].values))]['review_id'].values
    print(f"\tDone! Found {len(review_ids)} reviews.")
    print("Sample review IDs:", review_ids[:5])
    
    pickle.dump(review_ids, open('../data/yelp/restaurants_only/tmp/review_ids_for_df.pkl','wb'))

    return review_ids

def load_frame_lookups(path_to_framing_scores, debug):
    print(f"\nLoading in framing scores to create reviews df...")
    frame_lookup_paths = glob.glob(path_to_framing_scores+'/*.csv')
    if debug:
        print("\tDebug mode ON; loading only first 2 lookups")
        frame_lookup_paths = frame_lookup_paths[:2]
    print(f"\tFound {len(frame_lookup_paths)} lookups. Loading...")
    start = time.time()
    all_lookups = []
    for fname in tqdm(frame_lookup_paths):
        lookup = pd.read_csv(fname,index_col=0)
        all_lookups.append(lookup)

    master_lookup = pd.concat(all_lookups)
    print(f"\n\tDone! Created master lookup with shape {len(master_lookup)}. Elapsed time: {(time.time()-start)/60} minutes.")
    
    return master_lookup

def create_reviews_df(review_ids, raw_reviews, framing_scores_lookup, out_dir, debug):
    review_id2len = dict(zip(raw_reviews['review_id'], raw_reviews['len']))
    review_id2biz_id = dict(zip(raw_reviews['review_id'], raw_reviews['business_id']))
    
    feat_lists = glob.glob('feature_dicts/*.txt')
    feat_dict = {}
    for fname in feat_lists:
        feat_name = fname.split('/')[-1].split('.txt')[0]
        with open(fname,'r') as f:
            feat_dict[feat_name] = f.read().splitlines()
    avail_feats = [x.split('/')[-1].split('.txt')[0] for x in feat_lists]
    
    review_ids_avail = set([x.split('|')[0] for x in framing_scores_lookup.index])
#     print("Sample review IDs available:", list(review_ids_avail)[:5])
    
    review_ids_for_df = set(review_ids).intersection(review_ids_avail)
    missing_ids = set(review_ids).difference(review_ids_avail)
    print(f"Will create df using {len(review_ids_for_df)} review IDs after excluding {len(missing_ids)} reviews missing framing scores...")
    
    print("Now making reviews df...")
    per_review_df = defaultdict(list)                        
    for review_id in tqdm(review_ids_for_df):
        per_review_df['review_id'].append(review_id)
        per_review_df['review_len'].append(review_id2len[review_id])
        per_review_df['biz_id'].append(review_id2biz_id[review_id])
        
        # Add framing scores
        for feat in avail_feats:
            agg_score = 0
            agg_matches = Counter()
            for anchor_type in ['food','staff','venue']:
                res = framing_scores_lookup.loc[f"{review_id}|{feat}_{anchor_type}"]
                score, matches = int(res['score']), json.loads(res['matches'])
                per_review_df[f"{feat}_{anchor_type}_score"].append(score)
                per_review_df[f"{feat}_{anchor_type}_matches"].append(matches)
                agg_score += score
                if matches == -1:
                    matches = Counter()
                try:
                    agg_matches += matches
                except AttributeError:
                    print("error matches:", matches, type(matches), review_id)
            if agg_score == 0:
                assert agg_matches == Counter()
                agg_matches = -1
            per_review_df[f"{feat}_agg_score"].append(agg_score)
            per_review_df[f"{feat}_agg_matches"].append(json.dumps(agg_matches))
        
    per_review_df = pd.DataFrame(per_review_df)    
    print("\tCreated reviews df with shape:", per_review_df.shape)
    print(per_review_df.head())   
    savename = os.path.join(out_dir, 'per_reviews_df.csv')
    print("Saving reviews df to:", savename)
    per_review_df.to_pickle(savename)
    print("\tDone!")
    
    return per_review_df

def hydrate_reviews_with_biz_user_data(restaurants_df, reviews_df, out_dir):
    print("\nHydrating reviews df with user data...")
    review_id2user_id = pd.read_csv('data/yelp/review_id2user_id.csv')
    review_id2user_id = dict(zip(review_id2user_id['review_id'], review_id2user_id['user_id']))
    reviews_df['user_id'] = reviews_df['review_id'].apply(lambda x: review_id2user_id[x])
    print("\tDone!")
    
    print("\nHydrating reviews df with restaurant-related fields...")
    field2col_name = {'median_nb_income': 'Median household income in the past 12 months (2020 inflation-adjusted dollars)',
                      'nb_diversity': 'Race_Simpson_Diversity_Index',
                      'mean_star_rating': 'stars',
                      'price_point': 'price_level',
                      'nb_pct_asian': 'pct_asian',
                      'nb_pct_hisp': 'Percentage hispanic',
                      'cuisine_region': 'continents',
                      'cuisines': 'categories'}
        
    for field in tqdm(field2col_name):
        if field == 'cuisines':
            field_lookup = dict(zip(restaurants_df['business_id'], 
                                            restaurants_df[field2col_name[field]].apply(lambda x: set(x).intersection(TOP_CUISINES))))
        elif field == 'cuisine_region':
            field_lookup = dict(zip(restaurants_df['business_id'], 
                                            restaurants_df[field2col_name[field]].apply(lambda x: x[0] if len(set(x)) == 1
                                                                                  else 'fusion')))
        else:
            field_lookup = dict(zip(restaurants_df['business_id'], restaurants_df[field2col_name[field]]))
        
        reviews_df[f"biz_{field}"] = reviews_df['biz_id'].apply(lambda x: field_lookup[x])
    print("\tDone! New reviews_df columns:", reviews_df.columns)
    
    print("\nDropping nan neighborhood attributes...")
    reviews_df.dropna(subset=['biz_median_nb_income','biz_nb_diversity'], inplace=True)
    print(f"\tNew reviews_df length: {len(reviews_df)}")
    
    print("\nCuisine region distribution:")
    print(reviews_df['biz_cuisine_region'].value_counts())
    print("\nIndividual cuisine distribution:")
    for cuisine in TOP_CUISINES:
        print(cuisine, len(reviews_df.loc[reviews_df['biz_cuisines'].apply(lambda x: cuisine in x)]))
        
    savename = os.path.join(out_dir, 'per_reviews_df.csv')
    print(f"\nSaving hydrated df to: {savename}...")
    reviews_df.to_pickle(savename)
    print("\tDone!")
    
    return reviews_df
    
def main(path_to_enriched_df, path_to_raw_reviews, path_to_framing_scores, out_dir, debug):
    restaurants = prep_census_enriched_df(path_to_enriched_df)
    filtered_restaurants = filter_businesses_for_regression(restaurants)
    raw_reviews = load_raw_reviews(path_to_raw_reviews)
    review_ids = get_filtered_business_reviews(filtered_restaurants, raw_reviews)
    master_frame_lookup = load_frame_lookups(path_to_framing_scores, debug)
    reviews_df = create_reviews_df(review_ids, raw_reviews, master_frame_lookup, out_dir, debug)
    hydrated_reviews_df = hydrate_reviews_with_biz_user_data(filtered_restaurants, reviews_df, out_dir)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_enriched_df', type=str, default='../data/yelp/census_enriched_business_data.csv',
                        help='where to read in census enriched dataframe from')
    parser.add_argument('--path_to_raw_reviews', type=str, default='../data/yelp/restaurants_only/restaurant_reviews_df.csv',
                        help='where to read in raw reviews dataframe from')
    parser.add_argument('--path_to_framing_scores', type=str, default='../data/yelp/restaurants_only/agg_frame_lookups',
                        help='where to read in framing scores from')
    parser.add_argument('--out_dir', type=str, default='../data/yelp/restaurants_only',
                        help='directory to save output to')
    parser.add_argument('--debug', action='store_true',
                        help='whether to run on subset of data for debugging purposes')
    args = parser.parse_args()
    if not args.debug:
        print("\n******WARNING****** DEBUG MODE OFF!")
    else:
        print("\nRunning in debug mode; will load only 2 framing score lookups and limit dataframe creation to reviews for which those framing scores are available.")
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    main(args.path_to_enriched_df, args.path_to_raw_reviews, args.path_to_framing_scores, args.out_dir, args.debug)
    