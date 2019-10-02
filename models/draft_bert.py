import torch
from torch.functional import F
from enum import Enum
import numpy as np


class DraftBertTasks(Enum):
    DRAFT_PREDICTION = 1
    DRAFT_MATCHING = 2


class DraftBert(torch.nn.Module):
    def __init__(self, embedding_dim, ff_dim, n_head, n_encoder_layers, n_heros, out_ff_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.n_encoder_layers = n_encoder_layers
        self.n_heros = n_heros

        self.encoder_layer = torch.nn.TransformerEncoderLayer(embedding_dim, n_head, dim_feedforward=ff_dim)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)

        # Masked output layers
        self.dense_layer = torch.nn.Linear(embedding_dim, out_ff_dim)
        self.layer_norm = torch.nn.LayerNorm(out_ff_dim)
        self.output_layer = torch.nn.Linear(out_ff_dim, n_heros)

        dictionary_size = n_heros + 1 + 1  # + 1 for CLS token and + 1 for MASK
        self.PADDING_IDX = dictionary_size - 1
        self.hero_embeddings = torch.nn.Embedding(dictionary_size, embedding_dim, padding_idx=self.PADDING_IDX)

    def forward(self, src: torch.LongTensor, mask: torch.BoolTensor):
        """

        :param src: shape (batch_size, seq_length, 1) a sequence of hero_ids representing a draft
        :param tgt: shape (batch_size, seq_length, 1) a sequence of hero_ids representing a draft
        :param mask: shape (batch_size, seq_length) a boolean sequence of which indexes in the draft should be masked
        :return: shape (batch_size, seq_length, embedding_dim) encoded representation of the sequence
        """

        # First, we encode the src sequence into the latent hero representation
        src[mask] = self.PADDING_IDX  # Set the masked values to the embedding pad idx
        src = self.hero_embeddings(src)

        # Encoder expects shape (seq_length, batch_size, embedding_dim)
        src = src.permute(1, 0, 2)
        # Then we pass it through the encoder stack
        # out = self.encoder(src, src_key_padding_mask=mask, src_mask)
        out = self.encoder(src)
        # Encoder outputs shape (seq_length, batch_size, embedding_dim)
        out = out.permute(1, 0, 2)
        return out

    def get_masked_output(self, x):
        x = self.dense_layer(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x

    def _gen_random_masks(self, x: torch.LongTensor, pct=0.1):
        """

        :param x: shape (batch_size, sequence_length, 1)
        :param pct:
        :return:
        """
        n_masked_idx = int(x.shape[1] * pct)
        mask = np.append([1] * n_masked_idx, [0] * (x.shape[1] - n_masked_idx))
        mask = [np.random.permutation(mask) for _ in range(x.shape[0])]
        mask = torch.BoolTensor(mask)
        return mask

    def fit(self, src: torch.LongTensor, tgt: torch.LongTensor, task: DraftBertTasks, **train_kwargs):
        """

        :param src: shape (N, sequence_length, 1) where N is the size of the dataset
        :param tgt: shape (N, sequence_length, 1) where N is the size of the dataset
        :param task: Either draft prediction (fill in masked values) or draft matching (are these 2 teams from the same draft)
        :param train_kwargs: lr, batch_size, steps
        :return:
        """
        if isinstance(src, (list, np.ndarray)):
            src = torch.LongTensor(src)
        if isinstance(tgt, (list, np.ndarray)):
            tgt = torch.LongTensor(tgt)
        # Don't forget to set to train mode
        self.train()
        # self.cuda()
        lr = train_kwargs.get('lr', 0.001)
        batch_size = train_kwargs.get('batch_size', 512)
        steps = train_kwargs.get('steps', 100)
        mask_pct = train_kwargs.get('mask_pct', 0.1)

        if task == DraftBertTasks.DRAFT_PREDICTION:
            opt = torch.optim.Adam(self.parameters())
            N = src.shape[0]
            loss = torch.nn.CrossEntropyLoss()
            for step in range(steps):
                opt.zero_grad()
                idxs = np.random.choice(N, batch_size)
                src_batch, tgt_batch = src[idxs], tgt[idxs]
                masks = self._gen_random_masks(src_batch, mask_pct)

                src_batch = src_batch.cuda()
                masks = masks.cuda()

                out = self.forward(src_batch, masks)  # -> shape (batch_size, sequence_length, embedding_dim)
                to_predict = out[masks]
                # pred = to_predict.matmul(self.hero_embeddings.weight.T)
                pred = self.get_masked_output(to_predict)
                tgt_batch = tgt_batch[masks].cuda()
                batch_loss = loss(pred, tgt_batch)
                batch_loss.backward()
                opt.step()

                batch_acc = (pred.detach().cpu().numpy().argmax(1) == tgt_batch.detach().cpu().numpy()).astype(int).mean()
                print(f'Step: {step}, Loss: {batch_loss}, Acc: {batch_acc}')

    def predict(self, src: torch.LongTensor, mask: torch.BoolTensor, task: DraftBertTasks):
        if isinstance(src, (list, np.ndarray)):
            src = torch.LongTensor(src)
        if isinstance(mask, (list, np.ndarray)):
            mask = torch.BoolTensor(mask)
        self.train(False)
        if task == DraftBertTasks.DRAFT_PREDICTION:
            src = src.cuda()
            mask = mask.cuda()
            out = self.forward(src, mask)  # -> shape (batch_size, sequence_length, embedding_dim)
            to_predict = out[mask]
            pred = self.get_masked_output(to_predict)

            # pred = self.dense_layer(to_predict)
            return pred