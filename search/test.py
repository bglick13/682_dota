import sys, os, pickle
import numpy as np

from IPython import embed

sys.path.append(os.path.join('..'))

dt = pickle.load( open('../data/self_play/selfplay_1573193772.8818147.pickle', mode='rb'), encoding='bytes')

embed()