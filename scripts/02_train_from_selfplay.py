import os
import pickle

from torch import load, device, save

from models.draft_bert import DraftBert, SelfPlayDataset


def run(memory_file, ):
    BATCH_SIZE = 2048
    N_STEPS = 1000

    mems = []
    for mem in os.listdir(memory_file):
        if mem.endswith('pickle'):
            with open(os.path.join(memory_file, mem), 'rb') as f:
                mem = pickle.load(f)
            mems.append(mem)

    model: DraftBert = load('../weights/final_weights/train_from_selfplay_2.torch')
    with open('../weights/kmeans.pickle', 'rb') as f:
        clusterizer = pickle.load(f)

    dataset = SelfPlayDataset(*mems, clusterizer=clusterizer, le=model.le)
    total_points_sampled = BATCH_SIZE * N_STEPS
    EPOCHS = total_points_sampled // len(dataset)
    print(f'Dataset size: {len(dataset)}, Epochs: {EPOCHS}')

    model.cuda()
    model.train()
    model.matching_output.requires_grad = False
    model.masked_output.requires_grad = False
    model.train_from_selfplay(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, steps=N_STEPS, print_iter=10)
    save(model, os.path.join(memory_file, 'new_model.torch'))


if __name__ == '__main__':
    run('../data/self_play/memories_for_train_3')