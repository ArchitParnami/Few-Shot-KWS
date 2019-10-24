import torch

def convert_dict(k, v):
    return { k: v }

class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k,v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data

class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class EpisodicSpeechBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, 
                include_silence=False, include_unknown=False):
        self.n_classes = n_classes
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.include_silence = include_silence
        self.include_unknown = include_unknown
        self.skip = 0
        if include_silence:
            self.skip += 1
        if include_unknown:
            self.skip += 1

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            selected = torch.randperm(self.n_classes - self.skip)[:self.n_way - self.skip]

            if self.include_silence:
                silence_class = torch.tensor([self.n_classes - 2])
                selected = torch.cat((selected, silence_class))
            if self.include_unknown:
                unknown_class = torch.tensor([self.n_classes - 1])
                selected = torch.cat((selected, unknown_class))

            yield selected[torch.randperm(self.n_way)]
            