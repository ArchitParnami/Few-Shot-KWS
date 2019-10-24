import ast
import sys
import matplotlib.pyplot as plt
import os
import json

def plot_results(result_dir):
    train_acc = []; train_loss = []
    val_acc = []; val_loss = []
    result_file = os.path.join(result_dir, 'trace.txt')
    opt_file = os.path.join(result_dir, 'opt.json')
    opt = {}
    #get the options
    with open(opt_file, 'r') as rf:
        line = rf.readline()
        opt = json.loads(line)

    #get the results
    val_every = 1
    with open(result_file, 'r') as rf:    
        for line in rf:
            result = ast.literal_eval(line)
            train_epoch = result['epoch']
            train_acc.append(result['train']['acc'])
            train_loss.append(result['train']['loss'])
            if train_epoch % val_every == 0:
                val_acc.append(result['val']['acc'])
                val_loss.append(result['val']['loss'])

    train_epochs = range(1, len(train_acc)+1)
    val_epochs = range(val_every, len(train_acc)+1, val_every)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(train_epochs, train_loss, label='train')
    plt.plot(val_epochs, val_loss, label='val')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    data_opt = 'Way,Shot,Query = (Train:[{},{},{}], Val:[{} {} {}])'.format( 
        opt['data.way'], opt['data.shot'], opt['data.query'],
        opt['data.test_way'], opt['data.test_shot'], opt['data.test_query'])

    best_loss = min(val_loss)
    best_index = val_loss.index(best_loss)
    best_epoch = val_epochs[best_index]
    min_loss = 'Best Loss = (Train : {:.2}, Val: {:.2})'.format(train_loss[best_epoch-1], best_loss)
    max_acc = 'Best Accuracy = (Train : {:.2}, Val: {:.2})'.format(train_acc[best_epoch-1], val_acc[best_index])
    title = '{}\n{}\n{}'.format(data_opt, min_loss, max_acc)
    plt.title(title)
    plt.plot([best_epoch, best_epoch],[train_loss[best_epoch-1],best_loss], '--bo')

    plt.subplot(2,1,2)
    plt.plot(train_epochs, train_acc, label='train')
    plt.plot(val_epochs, val_acc, label='val')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    save_file = os.path.join(result_dir , 'plot.png')
    plt.savefig(save_file)


if __name__ == '__main__':
    result_dir = sys.argv[1]
    experiments = os.listdir(result_dir)
    for exp in experiments:
        exp_path = os.path.join(result_dir, exp)
        if os.path.isdir(exp_path):
            plot_results(exp_path)

    

