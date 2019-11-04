import numpy as np
import pandas as pd
import pickle


def reconcile_memory(memory):
    all_outcomes = []
    all_actions = []
    all_states = []
    for m in memory:
        v = m['all_values']
        s = m['all_states']
        a = m['all_actions']
        missing = len(s) - len(v)
        print(missing)
        for _ in range(missing):
            v.append(v[0])
        all_outcomes.extend(v)
        all_actions.extend(m['all_actions'])
        all_states.extend(m['all_states'])
    return all_states, all_actions, all_outcomes


def run(memory_file, ):
    with open(memory_file, 'rb') as f:
        mem = pickle.load(f)

    # TODO: I think we can probably convert this to the captiansmodepretrain dataset format and reuse a bunch of code
    states, actions, values = reconcile_memory(mem)