# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import csv
import json
import jsonlines
import os

import datasets
from datasets import Features


_CITATION = """\
@inproceedings{Zhu2023FIREBALL,
title={{FIREBALL: A Dataset of Dungeons and Dragons Actual-Play with Structured Game State Information}},
author={Zhu, Andrew and Aggarwal, Karmanya and Feng, Alexander and Martin, Lara J. and Callison-Burch, Chris},
year={2023},
booktitle={Annual Meeting of the Association for Computational Linguistics (ACL)},
month={7},
url={https://aclanthology.org/2023.acl-long.229/},
address={Toronto, Canada},
pages={4171--4193},
publisher={ACL},
doi={10.18653/v1/2023.acl-long.229}
}
"""
_DESCRIPTION = """\
FIREBALL Dungeons & Dragons data with narrative and Avrae scripting commands.
"""
_HOMEPAGE = "https://github.com/zhudotexe/FIREBALL"
_LICENSE = "cc-by-4.0"
_URLS = {
    "FIREBALL": "https://huggingface.co/datasets/lara-martin/FIREBALL/raw/main/"
}

class Fireball(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="FIREBALL", version=VERSION),
    ]


    def _info(self):
        
        features = Features(
            {
                "speaker_id": datasets.Value('int64'),
                "before_utterances": datasets.Sequence(datasets.Value('string')), 
                'combat_state_before': [{
                    'name': datasets.Value(dtype='string'),
                    'hp': datasets.Value(dtype='string'),
                    'class': datasets.Value(dtype='string'),
                    'race': datasets.Value(dtype='string'),
                    'attacks': datasets.Value(dtype='string'),
                    'spells': datasets.Value(dtype='string'),
                    'actions': datasets.Value(dtype='string'),
                    'effects': datasets.Value(dtype='string'),
                    'description': datasets.Value(dtype='string'),
                    'controller_id': datasets.Value(dtype='string')
                     
                }], #list of dictionaries
                'current_actor': { 
                    'name': datasets.Value(dtype='string'),
                    'hp': datasets.Value(dtype='string'),
                    'class': datasets.Value(dtype='string'),
                    'race': datasets.Value(dtype='string'),
                    'attacks': datasets.Value(dtype='string'),
                    'spells': datasets.Value(dtype='string'),
                    'actions': datasets.Value(dtype='string'),
                    'effects': datasets.Value(dtype='string'),
                    'description': datasets.Value(dtype='string'),
                    'controller_id': datasets.Value(dtype='string')
                }, #dictionary
                'commands_norm': datasets.Value('string'),
                'automation_results': datasets.Value('string'),
                'caster_after': {
                    'name': datasets.Value(dtype='string'),
                    'hp': datasets.Value(dtype='string'),
                    'class': datasets.Value(dtype='string'),
                    'race': datasets.Value(dtype='string'),
                    'attacks': datasets.Value(dtype='string'),
                    'spells': datasets.Value(dtype='string'),
                    'actions': datasets.Value(dtype='string'),
                    'effects': datasets.Value(dtype='string'),
                    'description': datasets.Value(dtype='string'),
                    'controller_id': datasets.Value(dtype='string')
                }, #dictionary
                'targets_after': [{
                    'name': datasets.Value(dtype='string'),
                    'hp': datasets.Value(dtype='string'),
                    'class': datasets.Value(dtype='string'),
                    'race': datasets.Value(dtype='string'),
                    'attacks': datasets.Value(dtype='string'),
                    'spells': datasets.Value(dtype='string'),
                    'actions': datasets.Value(dtype='string'),
                    'effects': datasets.Value(dtype='string'),
                    'description': datasets.Value(dtype='string'),
                    'controller_id': datasets.Value(dtype='string')
                     
                }], #list of dictionaries
                'combat_state_after': [{
                    'name': datasets.Value(dtype='string'),
                    'hp': datasets.Value(dtype='string'),
                    'class': datasets.Value(dtype='string'),
                    'race': datasets.Value(dtype='string'),
                    'attacks': datasets.Value(dtype='string'),
                    'spells': datasets.Value(dtype='string'),
                    'actions': datasets.Value(dtype='string'),
                    'effects': datasets.Value(dtype='string'),
                    'description': datasets.Value(dtype='string'),
                    'controller_id': datasets.Value(dtype='string')
                     
                }], #list of dictionaries
                'after_utterances': datasets.Sequence(datasets.Value('string')), 
                'utterance_history': datasets.Sequence(datasets.Value('string')),
                'before_idxs': datasets.Sequence(datasets.Value('int16')),
                'before_state_idx': datasets.Value('int16'),
                'command_idxs': datasets.Sequence(datasets.Value('int16')),
                'after_state_idx': datasets.Value('int16'),
                'after_idxs': datasets.Sequence(datasets.Value('int16')),
                'embed_idxs': datasets.Sequence(datasets.Value('int16'))
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # based off of OSCAR - https://huggingface.co/datasets/oscar/blob/main/oscar.py
        url = _URLS[self.config.name]
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        file_list = dl_manager.download(url+"files.txt")
        with open(file_list) as f:
            data_filenames  = [line.strip() for line in f if line]
            data_urls = dl_manager.download([url+"filtered/"+data_filename for data_filename in data_filenames])
            # data_urls = dl_manager.download([url+"filtered/00068c6b03adc2c102756053cf6edd05.jsonl"])
        downloaded_files = dl_manager.download(data_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": downloaded_files
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath):
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        key = 0
        for file in filepath:
            with jsonlines.open(file) as f:
              for data in f:
                # Yields examples as (key, example) tuples
                yield key, {
                    "speaker_id": data["speaker_id"],
                    "before_utterances": data["before_utterances"], 
                    'combat_state_before': data['combat_state_before'],
                    'current_actor': data["current_actor"],
                    'commands_norm': data['commands_norm'],
                    'automation_results': data['automation_results'],
                    'caster_after': data['caster_after'],
                    'targets_after': data['targets_after'],
                    'combat_state_after': data['combat_state_after'],
                    'after_utterances': data['after_utterances'],
                    'utterance_history': data['utterance_history'],
                    'before_idxs': data['before_idxs'],
                    'before_state_idx': data['before_state_idx'],
                    'command_idxs': data['command_idxs'],
                    'after_state_idx': data['after_state_idx'],
                    'after_idxs': data['after_idxs'],
                    'embed_idxs': data['embed_idxs']
                }
                key+=1
