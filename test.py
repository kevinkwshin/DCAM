import argparse
parser = argparse.ArgumentParser(description='PVC-NET')
parser.add_argument('--path', required=True, help='어느 것을 요구하냐')
args = parser.parse_args()
path = args.path

from data import *
from main import *

test(path, True)