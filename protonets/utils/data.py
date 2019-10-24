import protonets.data
import os
from protonets.data.FewShotSpeechData import FewShotSpeechDataset
import torch
from protonets.utils import filter_opt
from protonets.data.base import EpisodicBatchSampler, SequentialBatchSampler, EpisodicSpeechBatchSampler

def load(opt, splits):
    if opt['data.dataset'] in ['googlespeech']:
        ds = loader(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds

def loader(opt, splits):

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        if opt['data.dataset'] == 'googlespeech':
            speech_args = filter_opt(opt, 'speech')
            data_dir = os.path.join(os.path.dirname(__file__), '../../data/speech_commands/core')
            class_file = os.path.join(os.path.dirname(__file__), '../../data/speech_commands/core', split + '.txt')
            ds = FewShotSpeechDataset(data_dir, class_file, n_support, n_query, opt['data.cuda'], speech_args)        
        
        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
          
            sampler = EpisodicSpeechBatchSampler(len(ds), n_way, n_episodes,
                include_silence=opt['speech.include_silence'],
                include_unknown=opt['speech.include_unknown'])

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
