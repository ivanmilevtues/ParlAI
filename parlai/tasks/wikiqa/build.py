#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/wikiqa/wikiqa.tar.gz',
        'wikiqa.tar.gz',
        '9bb8851dfa8db89a209480e65a3d8967d8bbdf94d5d17a364c0381b0b7609412',
    )
]


def create_fb_format(outpath, dtype, inpath):
    print('building fbformat:' + dtype)
    fout = open(os.path.join(outpath, dtype + '.txt'), 'w')
    with PathManager.open(inpath) as f:
        lines = [line.strip('\n') for line in f]
    lastqid, lq, ans, cands = None, None, None, None
    for i in range(2, len(lines)):
        l = lines[i].split('\t')
        lqid = l[0]  # question id
        if lqid != lastqid:
            if lastqid is not None:
                # save
                s = '1 ' + lq + '\t' + ans.lstrip('|') + '\t\t' + cands.lstrip('|')
                if (dtype.find('filtered') == -1) or ans != '':
                    fout.write(s + '\n')
            # reset
            cands = ''
            ans = ''
            lastqid = lqid
        lcand = l[5]  # candidate answer / sentence from doc
        lq = l[1]  # question
        llabel = l[6]  # 0 or 1
        if int(llabel) == 1:
            ans = ans + '|' + lcand
        cands = cands + '|' + lcand
    fout.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'WikiQA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)
        import shutil

        dpext = os.path.join(dpath, 'WikiQACorpus')
        generate_from_json('/local/home/imilev/agent-tool-testing/src/wikidata_test/labeled_data_filtered.json')
        shutil.copy('/local/home/imilev/ParlAI/custom_questions.tsv', os.path.join(dpext, 'WikiQA-custom.tsv'))
        shutil.copy('/local/home/imilev/ParlAI/custom_questions.txt', os.path.join(dpext, 'WikiQA-custom.txt'))
        shutil.copy('/local/home/imilev/ParlAI/custom_questions.ref', os.path.join(dpext, 'WikiQA-custom.ref'))

        create_fb_format(dpath, 'train', os.path.join(dpext, 'WikiQA-train.tsv'))
        create_fb_format(dpath, 'valid', os.path.join(dpext, 'WikiQA-dev.tsv'))
        create_fb_format(dpath, 'test', os.path.join(dpext, 'WikiQA-test.tsv'))
        create_fb_format(dpath, 'test-custom', os.path.join(dpext, 'WikiQA-custom.tsv'))
        create_fb_format(dpath, 'train-filtered', os.path.join(dpext, 'WikiQA-train.tsv'))
        create_fb_format(dpath, 'valid-filtered', os.path.join(dpext, 'WikiQA-dev.tsv'))
        create_fb_format(dpath, 'test-filtered', os.path.join(dpext, 'WikiQA-test.tsv'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)


def generate_from_json(json_file):
    import csv
    with open(json_file, 'r') as f:
        data = json.loads(f.read())
    # generate 3 files: .tsv, .txt and .ref

    with open('./custom_questions.tsv', 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(['QuestionID', 'Question', 'DocumentID', 'DocumentTitle', 'SentenceID', 'Sentence', 'Label'])
        for entry in data:
            tsv_writer.writerow([entry['question_id'], entry['question'], entry['document_id'], 
                                 entry['document_title'], entry['sentence_id'], entry['sentence'], 1 if entry['label'] else 0])

    with open('./custom_questions.txt', 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        for entry in data:
            tsv_writer.writerow([entry['question'], entry['sentence'], 1 if entry['label'] else 0])


    with open('./custom_questions.ref', 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter=' ', lineterminator='\n')
        for entry in data:
            tsv_writer.writerow([
                int(entry['question_id'].split('Q')[-1]),
                0,
                int(entry['sentence_id'].split('-')[-1]),
                1 if entry['label'] else 0])
