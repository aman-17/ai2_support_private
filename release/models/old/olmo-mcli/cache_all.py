from cached_path import cached_path
import sys

path = sys.argv[1]
rank0_path = path + "/rank0.pt"
cached_path(rank0_path)
