#!/usr/bin/env python
# coding: utf-8

import os, glob
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import time

feat_lists = glob.glob('feature_dicts/*.txt')
feat_dict = {}
for fname in feat_lists:
    feat_name = fname.split('/')[-1].split('.txt')[0]
    with open(fname,'r') as f:
        feat_dict[feat_name] = f.read().splitlines()
avail_feats = [x.split('/')[-1].split('.txt')[0] for x in feat_lists]
print("Available dictionaries for feature aggregation:", sorted(avail_feats))
anchor_type2anchors = None
NEGATIONS = {'not','no'}

def load_anchor_sets():
    print("\nLoading anchor sets...")
    with open('anchor_sets/food_anchors.txt','r') as f:
        food_anchors = set(f.read().splitlines())
    with open('anchor_sets/establishment_anchors.txt','r') as f:
        establishment_anchors = set(f.read().splitlines())
    with open('anchor_sets/waitstaff_anchors.txt','r') as f:
        service_anchors = set(f.read().splitlines())
    print(f"\tDone! Found {len(food_anchors)} food anchors, {len(service_anchors)} staff anchors, and {len(establishment_anchors)} venue anchors.")
    anchor_type2anchors = {
        'food': food_anchors, 'staff': service_anchors, 'venue': establishment_anchors}
    for anchor_type in anchor_type2anchors:
        print(f"\tSample {anchor_type} anchors: {np.random.sample(anchor_type2anchors[anchor_type], 5, replace=False)}")

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

def aggregate_features(lookup, out_dir, add_to_cache, debug):
    
    print("\nAggregating features into framing dimensions using dictionaries...")
    if add_to_cache:
        print("\tLooking for existing aggregated features to add to...")
        try:
            agg_feats_per_review = dill.load(os.path.join(out_dir, "aggregated_frames_lookup.dill"), 'rb')
            print(f"\t\tLoaded cached dict of len {len(agg_feats_per_review)}!")
        except FileNotFoundError:
            print("No existing feature file found, exiting.")
            # TO DO quit
    else:
        agg_feats_per_review = defaultdict(dict)
    
    for _, key in tqdm(enumerate(lookup)):
        review_id, biz_id = key.split('|')
        review_frames = frames_lookup[review_id]
        for feat in avail_feats:
            no_neg_agg_score = 0
            no_neg_agg_matches = Counter()
            for anchor_type in ['food','service','establishment']:
                anchor_frames = [x for x in review_frames
                                 if x[2].replace('_',' ') in anchor_type2anchors[anchor_type]
                                 or x[3].replace('_',' ') in anchor_type2anchors[anchor_type]]
                anchor_frames_no_neg = [x[1] for x in anchor_frames
                                        if len(set(x[0].split(',')).intersection(NEGATIONS)) == 0]
#                 full_res = score_dict_feat(anchor_frames_with_neg, feat=feat)
#                 full_score = full_res[0]
#                 full_matches = full_res[1]
#                 full_agg_score += full_score
#                 full_agg_matches += full_matches
                no_neg_res = _score_dict_feat(anchor_frames_no_neg, feat=feat)
                no_neg_score = no_neg_res[0]
                no_neg_matches = no_neg_res[1] 
                no_neg_agg_score += no_neg_score
                no_neg_agg_matches += no_neg_matches
                if no_neg_score == 0:
                    no_neg_matches = -1
                agg_feats_per_review[review_id][feat][anchor_type] = (no_neg_score, no_neg_matches)
            if no_neg_agg_score == 0:
                no_neg_agg_matches = -1
            agg_feats_per_review[review_id][feat]['agg'] = (no_neg_agg_score, no_neg_agg_matches)
        
        if debug and _ > 10:
            dill.dump(open(os.path.join(out_dir, "test_aggregated_frames_lookup.dill"), 'wb'))
            # TODO quit (to avoid overwriting existing output)
    
    dill.dump(open(os.path.join(out_dir, "aggregated_frames_lookup.dill"), 'wb'))
    
def main(path_to_lookup, out_dir, debug):
    load_anchor_sets()
    lookup = load_lookup(path_to_lookup)
    aggregate_features(lookup, out_dir)
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_lookup', type=str, default='../data/yelp/restaurants_only/frames_lookup.pkl',
                        help='where to read in feature lookup dict from')
    parser.add_argument('--out_dir', type=str, default='../data/yelp/restaurants_only',
                        help='directory to save output to')
    
    parser.add_argument('--text_fields', type=str, default='text',
                        help='column name(s) for text fields')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='batch size for spaCy')
    parser.add_argument('--start_batch_no', type=int, default=0, 
                        help='batch number to start at')
    
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
    