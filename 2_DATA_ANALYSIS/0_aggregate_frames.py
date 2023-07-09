#!/usr/bin/env python
# coding: utf-8

import os, glob
import json
import pickle#, dill
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import sys
import time

feat_lists = glob.glob('feature_dicts/*.txt')
feat_dict = {}
for fname in feat_lists:
    feat_name = fname.split('/')[-1].split('.txt')[0]
    with open(fname,'r') as f:
        feat_dict[feat_name] = f.read().splitlines()
avail_feats = [x.split('/')[-1].split('.txt')[0] for x in feat_lists]
print("\n\nAvailable dictionaries for feature aggregation:", sorted(avail_feats))
NEGATIONS = {'not','no'}

def load_anchor_sets():
    print("\nLoading anchor sets...")
    with open('../anchor_sets/food_anchors.txt','r') as f:
        food_anchors = set(f.read().splitlines())
    with open('../anchor_sets/establishment_anchors.txt','r') as f:
        establishment_anchors = set(f.read().splitlines())
    with open('../anchor_sets/waitstaff_anchors.txt','r') as f:
        service_anchors = set(f.read().splitlines())
    print(f"\tDone! Found {len(food_anchors)} food anchors, {len(service_anchors)} staff anchors, and {len(establishment_anchors)} venue anchors.")
    anchor_type2anchors = {
        'food': food_anchors, 'staff': service_anchors, 'venue': establishment_anchors}
    for anchor_type in anchor_type2anchors:
        print(f"\tSample {anchor_type} anchors: {np.random.choice(a=list(anchor_type2anchors[anchor_type]), size=5, replace=False)}")
        
    return anchor_type2anchors

def load_lookup(path_to_lookup):
    print("\nLoading feature lookup dict...")
    frames_lookup = pickle.load(open(path_to_lookup,'rb'))
    print(f"\tDone! Loaded dict of len {len(frames_lookup)}. Sample lookup keys: {list(frames_lookup.keys())[:3]}.")
    print(f"\tSample frames: {frames_lookup[list(frames_lookup.keys())[0]]}")
    return frames_lookup

def _score_dict_feat(frames,feat='filtered_liwc_posemo',return_matches=True,normalize=False):
    match_set = feat_dict[feat]
    if len(frames) == 0:
        score = 0
        counted_matches = Counter()
    else:
        counted_frames = Counter(frames)
        counted_matches = {x: counted_frames[x] for x in set(counted_frames.keys()).intersection(match_set)}
        if len(counted_matches) > 0:
            feat_total = sum(counted_matches.values())
            if normalize:
                total = sum(counted_frames.values())
                score = feat_total / total
            else:
                score = feat_total
        else:
            score = 0
            counted_matches = Counter()
    if return_matches:
        return score, counted_matches
    else:
        return score

def aggregate_features(lookup, anchor_dict, out_dir, add_to_cache, debug, save_every=50000):
    
    out_dir = os.path.join(out_dir, 'agg_frame_lookups')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    print("\nAggregating features into framing dimensions using dictionaries...")
    if add_to_cache:
        print("\tLooking for cached aggregated features to add to...")
        try:
            old_agg_feats_per_review = dill.load(open(os.path.join(out_dir, "aggregated_frames_lookup.dill"), 'rb'))
            print(f"\t\tLoaded cached dict of length {len(old_agg_feats_per_review)}!")
        except FileNotFoundError:
            print("No existing feature file found, exiting.")
            sys.exit()
    else:
        print("\tCreating agg. features from scratch...")
        old_agg_feats_per_review = {} 
    
    reviews_to_add = set(lookup.keys()).difference(old_agg_feats_per_review.keys())
    print(f"\tFound {len(reviews_to_add)} reviews to add to cache. Sample IDs: {list(reviews_to_add)[:3]}. Adding...")
    agg_feats_per_review = dict()
    for _, review_id in tqdm(enumerate(reviews_to_add)):
        review_frames = lookup[review_id]
        for feat in avail_feats:
            for anchor_type in anchor_dict:
                anchor_frames = [x for x in review_frames
                                 if x[2].replace('_',' ') in anchor_dict[anchor_type]
                                 or x[3].replace('_',' ') in anchor_dict[anchor_type]]
                anchor_frames_no_neg = [x[1] for x in anchor_frames
                                        if len(set(x[0].split(',')).intersection(NEGATIONS)) == 0]
                no_neg_res = _score_dict_feat(anchor_frames_no_neg, feat=feat)
                no_neg_score = no_neg_res[0]
                no_neg_matches = no_neg_res[1] 
                if no_neg_score == 0:
                    no_neg_matches = -1
                agg_feats_per_review[f"{review_id}|{feat}_{anchor_type}"] = {'score':no_neg_score, 'matches':json.dumps(no_neg_matches)}
        
        if debug and _ == 9:
            print("\tSaving test aggregated features dict...")
            pd.DataFrame(agg_feats_per_review).T.to_csv(os.path.join(out_dir, "test_aggregated_frames_lookup.csv"), index=True)
            print("\t\tDone!")
            sys.exit()
            
        if _ % save_every == 0:
            savename = os.path.join(out_dir, f"aggregated_frames_lookup_{_}.csv")
            print(f"\tSaving aggregated features dict to {savename}...")
            start_time = time.time()
            pd.DataFrame(agg_feats_per_review).T.to_csv(savename, index=True)
            print(f"\t\tDone! Elapsed time: {(time.time()-start_time)/60} minutes.")
            print(f"\tCreating new lookup batch...")
            agg_feats_per_review = dict()
            print("\t\tNew lookup length:", len(agg_feats_per_review))
    
    print("\tSaving aggregated features dict...")
    start_time = time.time()
    pd.DataFrame(agg_feats_per_review).T.to_csv(os.path.join(out_dir, f"aggregated_frames_lookup_{_}.csv"), index=True)
    print(f"\t\tDone! Elapsed time: {(time.time()-start_time)/60} minutes.")
    
def main(path_to_lookup, out_dir, add_to_cache, debug):
    anchors_dict = load_anchor_sets()
    lookup = load_lookup(path_to_lookup)
    aggregate_features(lookup, anchors_dict, out_dir, add_to_cache, debug)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_lookup', type=str, default='../data/yelp/restaurants_only/frames_lookup.pkl',
                        help='where to read in feature lookup dict from')
    parser.add_argument('--out_dir', type=str, default='../data/yelp/restaurants_only',
                        help='directory to save output to')
    parser.add_argument('--add_to_cache', action='store_true',
                        help='whether to build on cached aggregated features')
    parser.add_argument('--debug', action='store_true',
                        help='whether to run on subset of data for debugging purposes')
    args = parser.parse_args()
    if not args.debug:
        print("\n******WARNING****** DEBUG MODE OFF!")
    else:
        print("\nRunning in debug mode; will limit to aggregating features of first 10 reviews.")
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    main(args.path_to_lookup, args.out_dir, args.add_to_cache, args.debug)
    