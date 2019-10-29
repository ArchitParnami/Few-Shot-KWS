import ast
import sys
import os
import json
import pandas as pd

root_dir = ''


def read_opt(opt_file):
    with open(opt_file, 'r') as rf:
        line = rf.readline()
        opt = json.loads(line)
        return opt

def read_trace(trace_file, val_every=1):
    train_acc = []; train_loss = []
    val_acc = []; val_loss = []
    with open(trace_file, 'r') as rf:    
        for line in rf:
            result = ast.literal_eval(line)
            train_epoch = result['epoch']
            train_acc.append(result['train']['acc'])
            train_loss.append(result['train']['loss'])
            if train_epoch % val_every == 0:
                val_acc.append(result['val']['acc'])
                val_loss.append(result['val']['loss'])
    return {'train' : {'acc' : train_acc, 'loss' : train_loss},
            'val' : {'acc' : val_acc, 'loss' : val_loss, 'every' : val_every}}

def get_best_results(trace):
    val_epochs = range(trace['val']['every'], 
            len(trace['train']['acc'])+1, trace['val']['every'])
    min_loss_val = min(trace['val']['loss'])
    best_index = trace['val']['loss'].index(min_loss_val)
    best_epoch = val_epochs[best_index]
    min_loss_train = trace['train']['loss'][best_epoch-1]
    max_acc_train = trace['train']['acc'][best_epoch-1]
    max_acc_val = trace['val']['acc'][best_index]
    return {'train' : {'acc' : max_acc_train, 'loss' : min_loss_train},
            'val'   : {'acc' : max_acc_val, 'loss' :min_loss_val},
            'epoch' : best_epoch}

def join_results(timestamp, opt, result):
    row = [opt['data.way'], opt['data.test_way'], opt['data.shot'], 
           opt['data.test_shot'], opt['data.query'], opt['data.test_query'],
           opt['data.train_episodes'], opt['data.test_episodes'], 
           opt['speech.include_background'], opt['speech.include_silence'],
           opt['speech.include_unknown'], opt['train.epochs'], opt['train.learning_rate'], 
           opt['train.weight_decay'], result['train']['acc'], result['val']['acc'],
           timestamp]
    return row
    


def read_results(root_dir):
    dirs = sorted(os.listdir(root_dir))
    columns = ['train.way', 'test.way', 'train.shot', 'test.shot',
               'train.query', 'test.query', 'train.episodes', 'test.episodes',
               'background', 'silence', 'unknown', 'epochs', 'lr', 'wd', 
               'train.acc', 'val.acc', 'timestamp']
    df = pd.DataFrame(columns=columns)

    for i, result_dir in enumerate(dirs):
        result_path = os.path.join(root_dir, result_dir)
        timestamp = os.listdir(result_path)[0]
        opt =read_opt(os.path.join(result_path, timestamp, 'opt.json'))
        trace = read_trace(os.path.join(result_path, timestamp, 'trace.txt'))
        result = get_best_results(trace)
        row = join_results(timestamp, opt, result)
        df.loc[i] = row

    return df


if __name__ == '__main__':
    result_root = sys.argv[1]
    df = read_results(result_root)
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        df.to_csv(output_file + '.csv', index=None, header=True)
    else:
        print(df)
    
