import os
import sys
from eval import main
import argparse

def eval_model(model_path):
    args = {'model.model_path' : model_path}
    main(args)

def evaluate(root_dir):
    dirs = sorted(os.listdir(root_dir))
    for i, result_dir in enumerate(dirs):
        result_path = os.path.join(root_dir, result_dir)
        timestamp = os.listdir(result_path)[0]
        model_path = os.path.join(result_path, timestamp, 'best_model.pt')
        eval_model(model_path)

if __name__ == '__main__':
    result_root = sys.argv[1]
    evaluate(result_root)


