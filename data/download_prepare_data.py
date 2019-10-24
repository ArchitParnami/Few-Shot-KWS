import os
import sys
import urllib
import tarfile
import torchaudio
import glob
import random
import shutil

class FewShotSpeechDataDownloader(object):
    def __init__(self, data_url, dest_directory, sample_rate, 
                clip_duration_ms, speaker_limit, core_split,
                unknown_split):
        self.data_url = data_url
        self.dataset_path = dest_directory
        self.desired_samples = int(sample_rate * clip_duration_ms / 1000)
        self.keywords = None
        self.speaker_limit = speaker_limit
        self.core_words_path = os.path.join(self.dataset_path, 'core_words.txt')
        self.other_words_path = os.path.join(self.dataset_path, 'other_words.txt')
        self.core_dataset_path = os.path.join(self.dataset_path, 'core')
        self.unknown_dataset_path = os.path.join(self.dataset_path, '_unknown_')
        self.num_core_speakers = None
        self.num_unknown_speakers = None
        self.core_split = core_split
        self.unknown_split = unknown_split

    def get_keywords(self):
        if self.keywords is None:
            keywords = [dir_name for dir_name in os.listdir(self.dataset_path) 
                if os.path.isdir(os.path.join(self.dataset_path, dir_name)) and 
                not dir_name.startswith('_')]
            self.keywords = keywords
        
        return self.keywords


    def load_audio(self, audio_file, desired_samples):
        sound, sr = torchaudio.load(filepath=audio_file, normalization=True,
                                         num_frames=desired_samples)
        return sound, sr

    def maybe_download_and_extract_dataset(self):
        """Download and extract data set tar file.
        
        If the data set we're using doesn't already exist, this function
        downloads it from the TensorFlow.org website and unpacks it into a
        directory.

        If the data_url is none, don't download anything and expect the data
        directory to contain the correct files already.

        Args:
        data_url: Web location of the tar file containing the data set.
        dest_directory: File path to extract data to.
        """
        if not self.data_url:
            return
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        else:
            print('{} already exists. Please specify a new path'.format(self.dataset_path))
            exit()
        filename = self.data_url.split('/')[-1]
        filepath = os.path.join(self.dataset_path, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            except:
                print(
                    'Failed to download URL: %s to folder: %s', self.data_url, filepath)
                print(
                    'Please make sure you have enough free space and'
                    ' an internet connection')
                raise
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded %s (%d bytes)',
                                        filename, statinfo.st_size)
            tarfile.open(filepath, 'r:gz').extractall(self.dataset_path)

    def delete_smaller_files(self):
        keywords = self.get_keywords()
        count = 0
        for keyword in keywords:
            keyword_wavs = os.path.join(self.dataset_path, keyword, '*.wav')
            for wav_file in sorted(glob.glob(keyword_wavs)):
                sound, sr = self.load_audio(wav_file, self.desired_samples)
                if sound.shape[1] != self.desired_samples:
                    os.unlink(wav_file)
                    count += 1
        print("{} smaller files deleted".format(count))
    
    def group_by_author(self):
        print("Grouping wave files by author..")
        path = self.dataset_path
        items = os.listdir(path)
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                wav_files = os.listdir(item_path)
                ids = {}
                for wav_file in wav_files:
                    pos = wav_file.find('_nohash_')
                    if pos != -1:
                        id = wav_file[:pos]
                        if id not in ids:
                            ids[id] = []
                        ids[id].append(wav_file)
                for id in ids:
                    dir_path = os.path.join(item_path, id)
                    os.makedirs(dir_path)
                    for wav_file in ids[id]:
                        wav_file_path = os.path.join(item_path, wav_file)
                        new_wave_file_path = os.path.join(dir_path, wav_file)
                        os.rename(wav_file_path, new_wave_file_path)
    
    def keyword_stats(self):
        stats = []
        for word in self.keywords:
            word_path = os.path.join(self.dataset_path, word)
            speakers = os.listdir(word_path)
            speaker_samples = []
            for speaker in speakers:
                speaker_path = os.path.join(word_path, speaker)
                samples = os.listdir(speaker_path)
                speaker_samples.append(len(samples))
            
            num_speakers = len(speakers)
            min_samples = min(speaker_samples)
            max_samples = max(speaker_samples)
            mean_samples = sum(speaker_samples) / num_speakers
            stats.append((word, num_speakers, min_samples, max_samples, mean_samples))
        return stats

    def analyze_data(self):
        stats = self.keyword_stats()
        stats.sort(key=lambda stat: stat[1], reverse=True)
        header = '{:8s} {:7s} {:3s} {:3s} {:3s}'.format("Word", "Speakers", "Min", "Max", "Mean")
        print('\n' + header + '\n')

        core = []; other = []
        min_core = float('inf')
        min_other = float('inf')

        for stat in stats:
            word, num_speakers, min_samples, max_samples, mean_samples = stat
            if num_speakers >= self.speaker_limit:
                core.append(word + '\n')
                if num_speakers < min_core:
                    min_core = num_speakers
            else:
                other.append(word +  '\n')
                if num_speakers < min_other:
                    min_other = num_speakers

            out = '{:8s} {:7d} {:3d} {:3d} {:.2f}'.format(word, num_speakers, min_samples, max_samples, mean_samples)
            print(out)

        print()
        with open(self.core_words_path, 'w') as wf:
            wf.writelines(core)
            print('Words with speakers >= {} saved to file {}'.format(speaker_limit, self.core_words_path))
            print('Min core word speakers = ', min_core)
            self.num_core_speakers = min_core

        with open(self.other_words_path, 'w') as wf:
            wf.writelines(other)
            print('Other saved to {}'.format(self.other_words_path))
            print('Min other word speakers = ', min_other)
            self.num_unknown_speakers = min_other
        print()

    def filter_dataset(self, path, new_path, words, num_speakers):
        os.makedirs(new_path)
        for word in words:
            item_path = os.path.join(path, word)
            new_item_path = os.path.join(new_path, word)
            os.makedirs(new_item_path)
            speakers = os.listdir(item_path)
            random.shuffle(speakers)
            speakers = speakers[:num_speakers]
            for speaker in speakers:
                speaker_path = os.path.join(item_path, speaker)
                wav_files = os.listdir(speaker_path)
                random_wave_file = wav_files[random.randrange(len(wav_files))]
                shutil.move(os.path.join(speaker_path, random_wave_file), 
                os.path.join(new_item_path, speaker + '.wav'))
            shutil.rmtree(item_path)
    
    def uniform_data(self):
        print('Making data uniform with respect to number of samples per keyword')
        with open(self.core_words_path, 'r') as rf:
            core_words = [line.strip('\n') for line in rf.readlines()]
        with open(self.other_words_path, 'r') as rf:
            unknown_words = [line.strip('\n') for line in rf.readlines()]
        self.filter_dataset(self.dataset_path, self.core_dataset_path, 
            core_words, self.num_core_speakers)
        self.filter_dataset(self.dataset_path, self.unknown_dataset_path,
            unknown_words, self.num_unknown_speakers)

    def write_file(self, path, classes):
        with open(path, 'w') as wf:
            for word in classes:
                wf.write(word + '\n')
    
    def write_splits(self, train, val, test, dataset_path):
        self.write_file(os.path.join(dataset_path, 'train.txt'), train)
        self.write_file(os.path.join(dataset_path, 'test.txt'), test)
        self.write_file(os.path.join(dataset_path, 'val.txt'), val)
    
    def generate_core_split(self):
        classes = os.listdir(self.core_dataset_path)
        random.shuffle(classes)
        tr, vl, te = self.core_split
        train = classes[:-(vl+te)]
        testval = classes[-(vl+te):]
        test = testval[:te]
        val = testval[te:]
        self.write_splits(train, val, test, self.core_dataset_path)
    
    def generate_unknown_split(self):
        words = sorted([word for word in os.listdir(self.unknown_dataset_path) 
            if os.path.isdir(os.path.join(self.unknown_dataset_path, word))])
        d = {}
        for word in words:
            d[word] = sorted(os.listdir(os.path.join(self.unknown_dataset_path, word)))
            random.shuffle(d[word])
            d[word] = d[word][:self.num_unknown_speakers]
        tr, vl , te = self.unknown_split
        train_count = int(tr * self.num_unknown_speakers / 100)
        test_count = int(te * self.num_unknown_speakers / 100)
        val_count = self.num_unknown_speakers - (train_count + test_count)
        train = []; test = []; val = []
        for i in range(self.num_unknown_speakers):
            for word in words:
                if i < train_count:
                    train.append(word + '/' + d[word][i])
                elif i < train_count + val_count:
                    val.append(word + '/' + d[word][i])
                else:
                    test.append(word + '/' + d[word][i])
        self.write_splits(train, val, test, self.unknown_dataset_path)

    def generate_splits(self):
        print('Generating train, val and test splits')
        self.generate_core_split()
        self.generate_unknown_split()
    
    def remove_unwanted_files(self):
        print('Cleaning up.')
        files = ['LICENSE', 'README.md', 'testing_list.txt', 
        'validation_list.txt', 'core_words.txt', 'other_words.txt']
        for file_name in files:
            file_path = os.path.join(self.dataset_path, file_name)
            if os.path.exists(file_path):
                os.unlink(file_path)
        print("You may delete the tar file now.")

    def run(self):
        self.maybe_download_and_extract_dataset()
        self.delete_smaller_files()
        self.group_by_author()
        self.analyze_data()
        self.uniform_data()
        self.generate_splits()
        self.remove_unwanted_files()
        print('All done.')


if __name__ == '__main__':
    data_url =  'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    dataset_path = os.path.join(os.curdir, 'speech_commands')
    sample_rate = 16000
    clip_duration_ms = 1000
    speaker_limit = 1000
    core_split = (24,3,3) # (train, val, test)
    unknown_split = (60, 20, 20) # in percentage

    downloader = FewShotSpeechDataDownloader(data_url, dataset_path,
        sample_rate, clip_duration_ms, speaker_limit, core_split, unknown_split)
    downloader.run()
    

