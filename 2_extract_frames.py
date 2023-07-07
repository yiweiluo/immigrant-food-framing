#!/usr/bin/env python
# coding: utf-8

import os, glob
import pickle
import pandas as pd
import argparse
from collections import Counter, defaultdict
import spacy
from spacy.tokens import DocBin, Doc
import time
from tqdm import tqdm, trange

spacy.prefer_gpu()
nlp_spacy = spacy.load("en_core_web_lg")

# with open('../anchor_sets/food_anchors.txt','r') as f:
#     food_anchors = set(f.read().splitlines())
# with open('../anchor_sets/establishment_anchors.txt','r') as f:
#     establishment_anchors = set(f.read().splitlines())
# with open('../anchor_sets/waitstaff_anchors.txt','r') as f:
#     service_anchors = set(f.read().splitlines())

def flatten(l):
    return [item for sublist in l for item in sublist]

def red_str(s):
    return f"\x1b[31m{s}\x1b[0m"

def load_raw_df(path_to_df):
    print("\nLoading in raw reviews...")
    file_sep = ',' if path_to_df.endswith('.csv') else '\t'
    df = pd.read_csv(path_to_df, sep=file_sep)
    print("\n\tRead in df with shape:", df.shape)
    if 'og_index' not in df.columns:
        print("\tNo OG index found; adding original index to keep track of batches...")
        df['og_index'] = list(range(len(df)))
    print("\tDone!")
    return df
          
def batch_df(df, batch_size):
    df['batch_no'] = [int(x/batch_size) for x in df.index]
    batch_size_counts = Counter(df['batch_no'].value_counts())
    remainder = min(batch_size_counts.keys())
    print(f"\nAssigned {len(df)} texts into {batch_size_counts[batch_size]} batches of size {batch_size} and {batch_size_counts[remainder]} batch of size {remainder}.")
    return df.groupby('batch_no')

def load_parsed_docs(path_to_parsed):
    print("\nLoading parsed review docs...")
    all_batches = glob.glob(path_to_parsed+'/*.spacy')
    doc_bin = DocBin().from_disk(all_batches[0])
    docs = list(doc_bin.get_docs(nlp_spacy.vocab))
    print(f"\tDone! Loaded {len(all_batches)} batches of docs of batch size {len(docs)}.")
    
    return all_batches, len(docs)

def get_attrs(doc, adj_tok, anchor=None):
    """Get text, lemma, ngram window, advmods etc. for a given (adj, anchor) framing instance"""
    
    # Recursively get all advmod/negation scoping over adj_tok
    adv_children = [c for c in adj_tok.children if c.dep_[-6:] == 'advmod' or c.dep_ == 'neg']
    if adj_tok.dep_ == 'acomp':
        adv_children.extend([c for c in adj_tok.head.children if c.dep_[-6:] == 'advmod' or c.dep_ == 'neg'])
    unvisited = adv_children.copy()
    while len(unvisited) > 0:
        curr_child = unvisited[0]
        new_children = [c for c in curr_child.children if c.dep_[-6:] == 'advmod' or c.dep_ == 'neg']
        adv_children.extend(new_children)
        unvisited.pop(0)
        unvisited.extend(new_children)
        
    # Map pronouns to anaphors
    mention2anaphor = {}
    for chain in doc._.coref_chains:
        #print(chain.index)
        #print(chain.mentions)
        #print(chain.most_specific_mention_index)
        for mention in chain.mentions:
            mention2anaphor[mention[0]] = chain.mentions[chain.most_specific_mention_index][0]
    
    out = {}
    out['mod'] = {
            'mod_ix': adj_tok.i,
            'mod_lemma': adj_tok.lemma_.lower(),
            'mod_token': adj_tok.text,
            'mod_advs': [x.lemma_.lower() for x in sorted(adv_children, key=lambda x: x.i)],
            'mod_bigram_left': '_'.join([doc[adj_tok.i-1].lemma_.lower(), doc[adj_tok.i].lemma_.lower()]) if adj_tok.i-1 > 0 else '',
            'mod_bigram_right': '_'.join([doc[adj_tok.i].lemma_.lower(), doc[adj_tok.i+1].lemma_.lower()]) if adj_tok.i+1 < len(doc) else '',
            'mod_trigram_left': '_'.join([doc[adj_tok.i-2].lemma_.lower(), doc[adj_tok.i-1].lemma_.lower(), doc[adj_tok.i].lemma_.lower()]) if adj_tok.i-2 > 0 else '',
            'mod_trigram_right': '_'.join([doc[adj_tok.i].lemma_.lower(), doc[adj_tok.i+1].lemma_.lower(), doc[adj_tok.i+2].lemma_.lower()]) if adj_tok.i+2 < len(doc) else '',
    }
    
    if anchor is None:
        out['anchor'] = {
            'anchor_ix': None,
            'anchor_lemma': '[none]',
            'anchor_token': '[none]',
            'full_anchor_ixs': set(),
            'full_anchor_lemmas': '[none]',
        }
    elif anchor.i in mention2anaphor: # is anaphora
        anaphor = doc[mention2anaphor[anchor.i]]
        out['anchor'] = {
            'anchor_ix': anchor.i,
            'anchor_lemma': anaphor.lemma_.lower(),
            'anchor_token': anaphor.text,
            'full_anchor_ixs':[c.i for c in anaphor.children if c.dep_ == 'compound'] + [anaphor.i],
            'full_anchor_lemmas':[c.lemma_.lower() for c in anaphor.children if c.dep_ == 'compound'] + [anaphor.lemma_.lower()]
        }
    else:
        out['anchor'] = {
            'anchor_ix': anchor.i,
            'anchor_lemma': anchor.lemma_.lower(),
            'anchor_token': anchor.text,
            'full_anchor_ixs':[c.i for c in anchor.children if c.dep_ == 'compound'] + [anchor.i],
            'full_anchor_lemmas':[c.lemma_.lower() for c in anchor.children if c.dep_ == 'compound'] + [anchor.lemma_.lower()]
        }
        
    return out
          
def extract_all_frames(
    batched_df,
    all_batches,
    path_to_parsed,
    out_dir,
    batch_size,
    from_cache,
    anchor_set='../anchor_sets/food_anchors.txt',  # set of terms for anchoring or path to one
    pos_tags={'ADJ'},                                                              
    verbose=False,
    start_batch_no=0,
    end_batch_no=None,
    print_every=500,
    save_every=100,
    guid='review_id'
):    
    """Extracts ngrams of (dependent, anchor) tuples from reviews."""
    
    print("\nExtracting frames from all parsed docs...")
    
    # I/O 
    out_dir_ = os.path.join(out_dir, 
                           f"dishnames={anchor_set.split('/')[-1].split('.')[0] if type(anchor_set)!=set else ','.join(anchor_set)}",
                           f"dependent-tags={','.join(sorted(list(pos_tags))) if type(pos_tags)==set else pos_tags}")
    if not os.path.exists(out_dir_):
        os.makedirs(out_dir_)
    savename = os.path.join(out_dir_, "ngrams_per_review.pkl")
    print("\tWill save extracted per review frames to:", savename)
    time.sleep(10)
    
    if from_cache:
        print("\nRetrieving (dependent, anchor) ngrams from cache...")
        lemmas_per_review = pickle.load(open(savename,'rb'))
        print("\tDone!")
    else:
        print("\nGetting (dependent, anchor) ngrams from scratch...")
        lemmas_per_review = defaultdict(list)
        
        # Specify POS and dependency criteria
        if pos_tags == 'any':
            pos_tag_set = {'ADJ','ADP','ADV','AUX','CONJ','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN',
                           'PUNCT','SCONJ','SYM','VERB','X','SPACE'}
        else:
            pos_tag_set = set(pos_tags)
        print("\nRestricting to dependents with any of the following POS tags:", pos_tags)
            
        # Iterate over processed batches
        print(f"\nRetrieving (dependent, anchor) ngrams...")
        max_batch_avail = max(batched_df.groups.keys())
        end_batch = min(end_batch_no, max_batch_avail)
        if end_batch_no > max_batch_avail:
            print(f"\tSpecified end_batch_no {end_batch_no} exceeds batches available, using max batch {max_batch_avail} as stopping point instead...")
        else:
            print(f"\tUsing specified end_batch_no {end_batch_no} as stopping point...")
        
        for batch_no in trange(start_batch_no, end_batch):
                
            out_fname = os.path.join(path_to_parsed, f'{batch_no}.spacy')
            reviews_batch = batched_df.get_group(batch_no)
            doc_bin = DocBin().from_disk(out_fname)
            docs = list(doc_bin.get_docs(nlp_spacy.vocab))
            
            # Iterate over processed reviews within each batch
            for row_ix,row in reviews_batch.iterrows():
                doc_ix = row['og_index'] % batch_size
                doc = docs[doc_ix]
                
                if (row_ix % print_every == 0):
                    print(f"\nOn doc {doc_ix} of batch {batch_no}")
                    print("Review text:", row['text'][:50])
                
                full_tokens = extract_frames_from_doc(doc, pos_tag_set=pos_tags)
                print("Extracted frames:", full_tokens)
                
                if row['og_index'] % print_every == 0:
                    all_tup_ixs = set([full_tokens[key]['mod']['mod_ix'] for key in full_tokens 
                                       if full_tokens[key] is not None])
                    all_tup_ixs |= set(flatten([full_tokens[key]['anchor']['full_anchor_ixs'] for key in full_tokens 
                                                if full_tokens[key] is not None]))
                    if 'business_id' in row:
                        print("Key in output:", f"{row[guid]}|{row['business_id']}")
                    else:
                        print("Key in output:", row[guid])
                    print("\nModified tokens:", ' '.join([red_str(tok.text)
                                                          if tok.i in all_tup_ixs
                                                          else tok.text
                                                          for tok in doc]))
                    time.sleep(5)
                
                if 'business_id' in row:
                    lemmas_per_review[f"{row[guid]}|{row['business_id']}"] = full_tokens
                else:
                    lemmas_per_review[f"{row[guid]}"] = full_tokens
                #break
            
            if (end_batch_no is not None) and (batch_no == end_batch_no):
                break
                
            if batch_no % save_every == 0:
                # Save lemmas
                print(f"\nSaving ngrams to {savename}...")
                pickle.dump(lemmas_per_review, open(savename,'wb'))
                print("\tDone!") 
            
        # Save lemmas
        print(f"\nSaving ngrams to {savename}...")
        pickle.dump(lemmas_per_review, open(savename,'wb'))
        print("\tDone!") 
                
    return lemmas_per_review

def extract_frames_from_doc(doc, pos_tag_set={'ADJ'}, verbose=False):
    all_frames_dict = {} # keys are frame (adj) indices within doc
    
    for adj_tok in doc:
        if adj_tok.pos_ in pos_tag_set:

            nsubjs = [c for c in adj_tok.children if c.dep_[:5] == 'nsubj']
            
            if adj_tok.dep_[-3:] == 'mod': # if plain modifying adj
                attrs_dict = get_attrs(doc, adj_tok, anchor=adj_tok.head) 
                all_frames_dict[adj_tok.i] = attrs_dict
            elif adj_tok.dep_ == 'acomp': # if acomp
                cop_verb = adj_tok.head
                if cop_verb.dep_ == 'relcl':
                    attrs_dict = get_attrs(doc, adj_tok, anchor=cop_verb.head) 
                    all_frames_dict[adj_tok.i] = attrs_dict
                else:
                    cop_subjs = [x for x in cop_verb.children if x.dep_[:5] == 'nsubj']
                    if cop_verb.dep_ == 'conj':
                        conjunct = cop_verb.head
                        while conjunct.dep_ == 'conj':
                            conjunct = conjunct.head
                        cop_subjs.extend([x for x in conjunct.children if x.dep_[:5] == 'nsubj'])
                    if len(cop_subjs) > 0:
                        for nsubj in cop_subjs:
                            attrs_dict = get_attrs(doc, adj_tok, anchor=nsubj) 
                            all_frames_dict[adj_tok.i] = attrs_dict
                    else:
                        attrs_dict = None
                        all_frames_dict[adj_tok.i] = attrs_dict
            elif len(nsubjs) > 0: # if is complement to nsubj
                for nsubj in nsubjs:
                    attrs_dict = get_attrs(doc, adj_tok, anchor=nsubj) 
                    all_frames_dict[adj_tok.i] = attrs_dict
            elif adj_tok.dep_ == 'xcomp': # if xcomp
                attrs_dict = get_attrs(doc, adj_tok, anchor=None) 
                all_frames_dict[adj_tok.i] = attrs_dict
            elif adj_tok.dep_ == 'conj': # if conjunct
                conjunct = adj_tok.head
                while conjunct.dep_ == 'conj':
                    conjunct = conjunct.head
                attrs_dict = conjunct.i  # set placeholder; copy over attrs from conjunct index later
            else:
                attrs_dict = None
                all_frames_dict[adj_tok.i] = attrs_dict

            if (type(attrs_dict) == int):
                if (attrs_dict in all_frames_dict) and (all_frames_dict[attrs_dict] is not None):
                    res = get_attrs(doc, adj_tok, anchor=None)
                    res['anchor'] = all_frames_dict[attrs_dict]['anchor']
                    all_frames_dict[adj_tok.i] = res
                else:
                    all_frames_dict[adj_tok.i] = None

            if verbose and (all_frames_dict[adj_tok.i] is not None):
                print((all_frames_dict[adj_tok.i]['mod']['mod_advs'],all_frames_dict[adj_tok.i]['mod']['mod_lemma'],all_frames_dict[adj_tok.i]['anchor']['full_anchor_lemmas']))
    
    return all_frames_dict

# def extract_token_tuples(
#     guid,
#     all_ngrams,
#     anchors=None, # anchors to filter to
#     #keep_anchors=False, # whether to include anchor lemmas in the retrieved ngrams 
#     unigrams=True, # whether to return unigram dependents
#     bigrams=True, # whether to return bigram dependents
#     trigrams=False, # whether to return trigram dependents
#     verbose=False,
#     compounds_file='/u/scr/yiweil/w-nw-food/compounds.txt',
#     compounds_set={}
# ):    
#     #print(compounds_file is None)
#     if compounds_file is not None:
#         compounds_set = retrieve_compounds(compounds_file=compounds_file, verbose=False)
    
#     full_ngrams = all_ngrams[guid]
#     tokens = []
#     for ix in full_ngrams:
#         d = full_ngrams[ix].copy()
#         #print("d:", d)
#         if d['full_anchor_lemmas'] != None:
#             #if d['full_anchor_lemmas'].replace('_',' ') in anchors:
#             #    if (f"{d['mod_lemma']} {d['full_anchor_lemmas']}" not in compounds_set) and (f"{d['mod_token']} {d['full_anchor_lemmas']}" not in compounds_set):
#             if unigrams:
#                 tokens.append((d['mod_lemma'], d['full_anchor_lemmas'], d['mod_advs'], guid))
#             if bigrams:
#                 tokens.append((d['mod_bigram_left'], d['full_anchor_lemmas'], d['mod_advs'], guid))
#                 tokens.append((d['mod_bigram_right'], d['full_anchor_lemmas'], d['mod_advs'], guid))
#             if trigrams:
#                 tokens.append((d['mod_trigram_left'], d['full_anchor_lemmas'], d['mod_advs'], guid))
#                 tokens.append((d['mod_trigram_right'], d['full_anchor_lemmas'], d['mod_advs'], guid))
# #                 else:
# #                     if verbose:
# #                         print(f"`{d['mod_lemma']} {d['full_anchor_lemmas']}` is a compound; excluding.") 
#         else:
#             if full_ngrams[ix]['anchor_ix'] in full_ngrams:
#                 head_d = full_ngrams[full_ngrams[ix]['anchor_ix']].copy()
#                 #print("head d:", head_d)
#                 while head_d['full_anchor_lemmas'] == None:
#                     if head_d['anchor_ix'] in full_ngrams:
#                         head_d = full_ngrams[head_d['anchor_ix']].copy()
#                     else:
#                         break
#                 if head_d['full_anchor_lemmas'] != None:
# #                     if head_d['full_anchor_lemmas'].replace('_',' ') in anchors:
# #                         if (f"{d['mod_lemma']} {d['full_anchor_lemmas']}" not in compounds_set) and (f"{d['mod_token']} {d['full_anchor_lemmas']}" not in compounds_set):
#                     if unigrams:
#                         tokens.append((d['mod_lemma'], head_d['full_anchor_lemmas'], d['mod_advs'], guid))
#                     if bigrams:
#                         tokens.append((d['mod_bigram_left'], head_d['full_anchor_lemmas'], d['mod_advs'], guid))
#                         tokens.append((d['mod_bigram_right'], head_d['full_anchor_lemmas'], d['mod_advs'], guid))
#                     if trigrams:
#                         tokens.append((d['mod_trigram_left'], head_d['full_anchor_lemmas'], d['mod_advs'], guid))
#                         tokens.append((d['mod_trigram_right'], head_d['full_anchor_lemmas'], d['mod_advs'], guid))
# #                     else:
# #                         if verbose:
# #                             print(f"`{d['mod_lemma']} {d['full_anchor_lemmas']}` is a compound; excluding.") 
# #             else:
# #                 print(f"Head token of `{full_tokens[ix]['mod_lemma']}` not an anchor dependent from which to inherit full anchor lemma.")
#     #print("\nModified tokens:", tokens)
#     return tokens

def get_frame_tuples(doc_frames):
    """Assumes unigrams"""
    out = [(doc_frames[frame_ix]['mod']['mod_advs'], 
            doc_frames[frame_ix]['mod']['mod_lemma'], 
            doc_frames[frame_ix]['anchor']['anchor_lemma'], 
            doc_frames[frame_ix]['anchor']['full_anchor_lemmas']) 
           for frame_ix in doc_frames
           if doc_frames[frame_ix] is not None]
    
    return out

def create_frames_lookup(lemmas_per_review, out_dir, debug):
    print('\nCreating frames lookup...')
    
    guid = list(lemmas_per_review.keys())[0]
    D = defaultdict(list)
        
    for guid in tqdm(lemmas_per_review):
        res = get_frame_tuples(lemmas_per_review[guid])
        for tup_ in res:
            anchor = tup_[2]
            if '|' in guid:
                review_id, restaurant_id = guid.split('|')
            else:
                review_id = guid
                restaurant_id = None
            
            D[review_id].append((','.join(tup_[0]), tup_[1], tup_[2], '_'.join(tup_[3])))
        
        if debug:
            break
            print(D)
            
    print(f"\tDone! Saving lookup to {os.path.join(out_dir,'frames_lookup.pkl')}...")
    time.sleep(5)
    pickle.dump(D, open(os.path.join(out_dir,'frames_lookup.pkl'),'wb'))
    print("\t\tDone saving!")
    
    return D
    
def main(path_to_df, path_to_parsed, out_dir, guid, start_batch_no, end_batch_no, from_cache, debug):
    raw_df = load_raw_df(path_to_df)
    batched_df = batch_df(raw_df, batch_size)
    parsed_docs, batch_size = load_parsed_docs(path_to_parsed)
    lemmas_per_review = extract_all_frames(batched_df, parsed_docs, path_to_parsed, out_dir, batch_size, from_cache, 
                                           start_batch_no=start_batch_no, end_batch_no=end_batch_no, guid=guid)
    create_frames_lookup(lemmas_per_review, out_dir, debug)
    print("\n\nAll done!")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_raw', type=str, default='../data/yelp/restaurants_only/restaurant_reviews_df.csv',
                        help='where to read in raw reviews dataframe from')
    parser.add_argument('--path_to_parsed', type=str, default='../data/yelp/restaurants_only/spacy_processed',
                        help='where to read in parsed reviews from')
    parser.add_argument('--out_dir', type=str, default='../data/yelp/restaurants_only/',
                        help='directory to save output to')
    parser.add_argument('--guid', type=str, default='review_id',
                        help='field to use for GUID in raw data df')
#     parser.add_argument('--batch_size', type=int, default=5000,
#                         help='batch size for spaCy')
    parser.add_argument('--start_batch_no', type=int, default=0,
                        help='start batch index')
    parser.add_argument('--end_batch_no', type=int, default=1026,
                        help='end batch index (non-inclusive)')
    parser.add_argument('--from_cache', action='store_true',
                        help='whether to load extracted ngrams from cache')
    parser.add_argument('--debug', action='store_true',
                        help='whether to test frames_lookup.pkl')
    args = parser.parse_args()
        
    main(args.path_to_raw, args.path_to_parsed, args.out_dir, args.guid, args.batch_size, args.start_batch_no, args.end_batch_no, args.from_cache, args.debug)
    