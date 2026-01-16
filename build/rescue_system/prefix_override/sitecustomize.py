import sys
if sys.prefix == '/home/hansung/miniconda3/envs/kroc':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/hansung/kroc/src/rescue_system/install/rescue_system'
