#!/usr/bin/env python
# coding: utf-8

import os, glob
import pickle
import pandas as pd
import argparse
import spacy
from spacy.tokens import DocBin, Doc
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import re
import time

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('coreferee')

def load_df(path_to_df, debug, start_batch_no, end_batch_no, batch_size):
    file_sep = ',' if path_to_df.endswith('.csv') else '\t'
#     print(f"\nReading in lines {start_batch_no*batch_size} to {start_batch_no*batch_size+(end_batch_no-start_batch_no)*batch_size} of df...")
#     df = pd.read_csv(path_to_df, sep=file_sep, skiprows=range(1,start_batch_no*batch_size+1), nrows=(end_batch_no-start_batch_no)*batch_size)#, index_col=0)
    print(f"\nReading in df...")
    df = pd.read_csv(path_to_df, sep=file_sep)
    if 'og_index' not in df.columns:
        print("\tNo OG index found; adding original index to keep track of batches...")
        df['og_index'] = list(range(len(df)))
        
    print(f"\tDone! Start, end indices: ({df['og_index'].values[0]}, {df['og_index'].values[-1]})")
    return df

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

def batch_spacy_process(out_dir, df, start_batch_no, end_batch_no, batch_size, text_fields='text', debug=False, overwrite=False):
    
    if debug:
        print("\nDebug mode ON, will stop after first batch.")
        
    text_fields = text_fields.split(',')
    if len(text_fields) > 1:
        print("\nUsing concatenation of the strs associated with the following column names as text fields:", text_fields)
    else:
        print(f"\nUsing str associated with `{text_fields[0]}` as text field.")
    
    #batch_groups = batched_df.groupby('batch_no')
    for batch_no in trange(start_batch_no, end_batch_no, 1):
        
        out_fname = os.path.join(out_dir, f'{batch_no}.spacy')
        if os.path.exists(out_fname) and not overwrite:
            print(f"\t{out_fname} already exists; skipping current batch.")
            continue 
            
        batch = df.iloc[batch_no*batch_size:batch_no*batch_size+batch_size]
        if len(batch) == 0:
            print("\nRan out of input, terminating.")
            break
        print(f"\nProcessing batch {batch_no} of length {len(batch)}, with (start, end) indices = ({batch['og_index'].values[0]}, {batch['og_index'].values[-1]})...")
        time.sleep(5)

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
    
def main(path_to_dataset, out_dir, text_fields, batch_size, start_batch_no, end_batch_no, debug, overwrite):
    texts = load_df(path_to_dataset, debug, start_batch_no, end_batch_no, batch_size)
    reviews = batch_df(texts, batch_size)
    batch_spacy_process(out_dir, texts, start_batch_no, end_batch_no, batch_size, text_fields=text_fields, debug=debug, overwrite=overwrite)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_texts', type=str, default='../data/yelp/restaurants_only/restaurant_reviews_df.csv',
                        help='where to read in texts dataframe from')
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
    parser.add_argument('--overwrite', action='store_true',
                        help='whether to overwrite existing batches')
    args = parser.parse_args()
    if not args.debug:
        print("\n******WARNING****** DEBUG MODE OFF!")
    else:
        print("\nRunning in debug mode; will limit to processing first batch of texts with batch size of 10.")
        args.start_batch_no, args.end_batch_no, args.batch_size = 0, 1, 10
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    main(args.path_to_texts, args.out_dir, args.text_fields, args.batch_size, args.start_batch_no, args.end_batch_no, args.debug, args.overwrite)
    