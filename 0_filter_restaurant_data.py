#!/usr/bin/env python
# coding: utf-8

# Script for subsetting Yelp Academic Dataset business/review data to items from restaurants

import os
import pandas as pd
import argparse

def is_cat(cat_list, search_cats=None):
    """Returns whether a list of str categories contains one from a search set."""
    return len(set([x.strip().lower() for x in cat_list if type(x) == str])\
               .intersection(search_cats)) > 0
    
def load_business_data(path_to_data_dir):
    df = pd.read_json(os.path.join(path_to_data_dir,'yelp_academic_dataset_business.json'), lines=True)
    print(f"\nRead in businesses df with shape {df.shape}.")
    df['categories_list'] = df['categories'].apply(lambda x: x.split(',') if x != None else [x]) # convert str to list
    return df
    
def load_review_data(path_to_data_dir, debug=False, thresh=1000):
    print("\nReading in reviews...")
    if debug:
        first_N = ""
        with open(os.path.join(path_to_data_dir,'yelp_academic_dataset_review.json'), "r") as file:  
            for i in range(thresh):
                line = next(file).strip()
                first_N += line
                first_N += '\n'
        df = pd.read_json(first_N, lines=True)
    else:
        df = pd.read_json(os.path.join(path_to_data_dir,'yelp_academic_dataset_review.json'),
                                   lines=True)
    print(f"\tRead in reviews df with shape {df.shape}.")
    return df
    
def filter_businesses(business_df, filter_to_cats={'restaurants','food'}):
    print("\nFiltering businesses to restaurants...")
    filtered_df = business_df.loc[business_df['categories_list'].apply(lambda x: is_cat(x, search_cats=filter_to_cats)) == True]
    print(f"\tDone! Found {len(filtered_df)} restaurants.")
    return filtered_df

def get_restaurant_ids(restaurant_df):
    """Returns Yelp IDs associated with restaurants found in filtered dataframe."""
    return set(restaurant_df['business_id'])
    
def filter_reviews(reviews_df, restaurant_ids_set):
    print("\nFiltering reviews to restaurant reviews...")
    filtered_df = reviews_df.loc[reviews_df['business_id'].isin(restaurant_ids_set)]
    print(f"\tDone! Found {len(filtered_df)} restaurant reviews.")
    return filtered_df
    
def save_data(df, out_dir, fname):
    out_fname = os.path.join(out_dir, fname)
    print(f"\nSaving data to {out_fname}...")
    df.to_csv(os.path.join(out_dir, fname), index=False)
    print("\tDone!")
    
def main(path_to_dataset, out_dir, debug):
    business_data = load_business_data(path_to_dataset)
    restaurants = filter_businesses(business_data)
    save_data(restaurants, out_dir, 'restaurants_df.csv')
    review_data = load_review_data(path_to_dataset, debug=debug)
    restaurant_ids = get_restaurant_ids(restaurants)
    restaurant_reviews = filter_reviews(review_data, restaurant_ids)
    save_data(restaurant_reviews, out_dir, 'restaurant_reviews_df.csv')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data_dir', type=str, default='data/yelp',
                        help='directory to read in Yelp open dataset from')
    parser.add_argument('--out_dir', type=str, default='data/yelp/restaurants_only',
                        help='directory to save output filtered dataframes to')
    parser.add_argument('--debug', action='store_true',
                        help='whether to run on subset of data for debugging purposes')
    args = parser.parse_args()
    if not args.debug:
        print("\n******WARNING****** DEBUG MODE OFF!")
    else:
        print("\nRunning in debug mode; will limit to ingesting first 1000 reviews.")
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    main(args.path_to_data_dir, args.out_dir, args.debug)
    