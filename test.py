import argparse
parser = argparse.ArgumentParser(description='PVC-NET')
parser.add_argument('--ckpt_path', required=True, help='path to ckpt')
args = parser.parse_args()
ckpt_path = args.ckpt_path

from data import *
from main import *

test(ckpt_path, True)