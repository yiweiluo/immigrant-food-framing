#!/usr/bin/env python
# coding: utf-8

import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import argparse

# actual distribution of Yelp reviews
target_pcts_per_sentiment = {
    'Very positive':.454797,
    'Positive':.248251,
    'Neutral':.115715,
    'Negative':.098499,
    'Very negative':.082738
}

def strip_disclaimer(s):
    return '\n\n'.join([x for x in s.split('\n\n') 
                        if 'language model' not in x.lower() 
                        and 'openai' not in x.lower()])

def main(path_to_jsons, out_dir):
        
    collected = glob.glob(os.path.join(path_to_jsons, '*.json'))
    print(f"\nCollected {len(collected)} LLM review jsons.")
    print(collected[:3])
    print()
    
    # aggregate jsons
    print("Aggregating jsons...")
    json_list = []
    for fname in tqdm(collected):
        with open(fname, 'r') as f:
            obj = json.loads(f.read())
            json_list.append(obj)
    df = pd.DataFrame(json_list)
    print('\tShape of aggregated dataframe', df.shape)
    print()
    print(df.head())
    print(df.tail())
    print(df['prompt_ix'].value_counts()) 
    print(df.loc[df['prompt_ix']==0]['sentiment'].value_counts(normalize=True)) # verify target pcts
    print(df.loc[df['model']=='gpt-3.5-turbo-0613']['sentiment'].value_counts(normalize=True))
    
    # remove disclaimer text
    df['review_no_disclaimers'] = df['review'].apply(lambda x: strip_disclaimer(x))
    
    # remove errors
    df_no_err = df.loc[df['review_no_disclaimers']!='error']
    print('\nAfter removing API errors, resulting aggregate dataframe shape:', df_no_err.shape)
    print()
    
    # add GUID
    df_no_err['guid'] = range(len(df_no_err))

    # save
    print(f"Saving aggregated output to: {os.path.join(out_dir, 'gpt_new_reviews.csv')}")
    df_no_err.to_csv(os.path.join(out_dir, 'gpt_new_reviews.csv'))
    print("\tAll done!")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_jsons', type=str, default='../data/gpt_output',
                        help='directory containing GPT output')
    parser.add_argument('--out_dir', type=str, default='../data/llm',
                        help='directory to save output to')
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
            
    main(args.path_to_jsons, args.out_dir)
    