#!/usr/bin/env python
# coding: utf-8

import os, glob, json
import pickle
import pandas as pd
import argparse
from tqdm import tqdm, trange
from collections import Counter, defaultdict

REGIONS = {'us','europe','latin_america','asia'}
CHAIN_THRESHOLD = 1

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
    restaurant_data = pd.read_csv(path_to_enriched_df,index_col=0)
    print(f"Read in census enriched business data from {path_to_enriched_df} with {len(restaurant_data)} rows.")

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

def batch_df(df, batch_size):
    df['batch_no'] = [int(x/batch_size) for x in df.index]
    batch_size_counts = Counter(df['batch_no'].value_counts())
    remainder = min(batch_size_counts.keys())
    print(f"\nAssigned {len(df)} texts into {batch_size_counts[batch_size]} batches of size {batch_size} and {batch_size_counts[remainder]} batch of size {remainder}.")
    return df

def save_guid_batch_no_info(debug, out_dir, batched_df, batch_size, guid_str='review_id'):
    guid2batch_no = dict(zip(batched_df[guid_str], batched_df['batch_no']))
    batch_no2guids = {i: batched_df.iloc[range(i*batch_size,i*batch_size+batch_size)][guid_str].values for i in range(max(batched_df['batch_no'])+1)}
    
    if debug:
        print(guid2batch_no)
        print(batch_no2guids)
    
    pickle.dump(guid2batch_no, open(os.path.join(out_dir, 'guid2batch_no.pkl'),'wb'))
    pickle.dump(batch_no2guids, open(os.path.join(out_dir, 'batch_no2guids.pkl'),'wb'))
    print(f"\nCreated lookups from GUIDs (field={guid_str}) to batch number and vice versa:", glob.glob(os.path.join(out_dir, '*.pkl')))
    
def strip_punc(raw_text):
    return re.sub(r'[^\w\s]','',raw_text)

def spacy_process(raw_text):
    doc = nlp(raw_text)
    return doc

def batch_spacy_process(out_dir, df, start_batch_no, end_batch_no, batch_size, text_fields='text', debug=False):
    
    if debug:
        print("\nDebug mode ON, will stop after first batch.")
        
    text_fields = text_fields.split(',')
    if len(text_fields) > 1:
        print("\nUsing concatenation of the strs associated with the following column names as text fields:", text_fields)
    else:
        print(f"\nUsing str associated with `{text_fields[0]}` as text field.")
    
    #batch_groups = batched_df.groupby('batch_no')
    for batch_no in trange(start_batch_no, end_batch_no, 1):
        batch = df.iloc[(batch_no-start_batch_no)*batch_size:(batch_no-start_batch_no+1)*batch_size]
        if len(batch) == 0:
            print("\nRan out of input, terminating.")
            break
        print(f"\nProcessing batch {batch_no} of length {len(batch)}, with (start, end) indices = ({batch['og_index'].values[0]}, {batch['og_index'].values[-1]})...")
        time.sleep(20)

        doc_bin = DocBin(attrs=["ORTH", "TAG", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE", "ENT_KB_ID", "LEMMA", "MORPH", "POS"], store_user_data=True)
        for row_ix, row in tqdm(batch.iterrows()):
            text = ". ".join([row[tf] for tf in text_fields if type(row[tf])==str])
            if (type(text) == str) and (len(strip_punc(text)) > 0):
                doc = spacy_process(text)
            else:
                doc = Doc(nlp.vocab)
            if debug:
                print('\nRaw text:', text)
                print('\n\tProcessed lemmas:', ' '.join([tok.lemma_ for tok in doc]))
            doc_bin.add(doc)

        out_fname = os.path.join(out_dir, f'{batch_no}.spacy')
        print(f"\tDone processing! Saving to disk at: {out_fname}...")
        doc_bin.to_disk(out_fname)
        print("\t\tDone!")

        if debug:
            print("\nTest deserialization...")
            #nlp = spacy.blank("en")
            doc_bin = DocBin().from_disk(out_fname)
            docs = list(doc_bin.get_docs(nlp.vocab))
            for tok in docs[0]:
                print(tok.text, tok.lemma_, tok.head.text, tok.head.dep_, tok.pos_, tok.ent_type_)
            print()
            print("Coref results:", docs[0]._.coref_chains)
            break
    
def main(path_to_enriched_df, out_dir, text_fields, batch_size, start_batch_no, end_batch_no, debug):
    restaurants = prep_census_enriched_df(path_to_enriched_df)
    #reviews = batch_df(texts, batch_size)
    #batch_spacy_process(out_dir, texts, start_batch_no, end_batch_no, batch_size, text_fields=text_fields, debug=debug)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_enriched_df', type=str, default='../data/yelp/census_enriched_business_data.csv',
                        help='where to read in census enriched dataframe from')
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
        
    main(args.path_to_enriched_df, args.out_dir, args.text_fields, args.batch_size, args.start_batch_no, args.end_batch_no, args.debug)
    