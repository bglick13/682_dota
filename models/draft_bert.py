import torch
from torch.functional import F
from enum import Enum
import numpy as np
from tqdm import tqdm


class DraftBertTasks(Enum):
    DRAFT_PREDICTION = 1
    DRAFT_MATCHING = 2
    WIN_PREDICTION = 3


class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class DraftBert(torch.nn.Module):
    def __init__(self, embedding_dim, ff_dim, n_head, n_encoder_layers, n_heros, out_ff_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.n_encoder_layers = n_encoder_layers
        self.n_heros = n_heros

        self.encoder_layer = torch.nn.TransformerEncoderLayer(embedding_dim, n_head, dim_feedforward=ff_dim, dropout=0.2)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)

        # Masked output layers
        self.dense_layer = torch.nn.Linear(embedding_dim, out_ff_dim)
        self.layer_norm = torch.nn.LayerNorm(out_ff_dim)
        self.output_layer = torch.nn.Linear(out_ff_dim, n_heros)

        # Matching classifier layer
        self.matching_layer = torch.nn.Linear(embedding_dim, 2)

        dictionary_size = n_heros + 1 + 1 + 1  # + 1 for CLS token, + 1 for SEP token, and + 1 for MASK
        self.PADDING_IDX = dictionary_size - 1
        self.CLS_IDX = dictionary_size - 2
        self.hero_embeddings = torch.nn.Embedding(dictionary_size, embedding_dim, padding_idx=self.PADDING_IDX)
        self.pe = PositionalEncoding(embedding_dim, 0, max_len=13)

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
        src = src + np.sqrt(self.embedding_dim)
        src = self.pe(src)

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

    def get_matching_output(self, x):
        return self.matching_layer(x)

    def _gen_random_masks(self, x: torch.LongTensor, pct=0.1):
        """

        :param x: shape (batch_size, sequence_length, 1)
        :param pct:
        :return:
        """
        n_masked_idx = int((x.shape[1] - 3) * pct)
        mask = np.append([1] * n_masked_idx, [0] * (x.shape[1] - n_masked_idx - 3))
        mask = np.array([np.random.permutation(mask) for _ in range(x.shape[0])])
        zeros = np.zeros((x.shape[0], 1))
        mask = np.hstack((zeros,
                          mask[:, :5],
                          zeros,
                          mask[:, 5:],
                          zeros))

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
        print_iter = train_kwargs.get('print_iter', 100)
        save_iter = train_kwargs.get('save_iter', 100000)

        if task == DraftBertTasks.DRAFT_PREDICTION:
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            N = src.shape[0]
            mask_loss = torch.nn.CrossEntropyLoss(reduction='mean')
            matching_loss = torch.nn.CrossEntropyLoss(reduction='mean')
            for step in tqdm(range(steps)):
                opt.zero_grad()
                idxs = np.random.choice(N, batch_size)

                # Sample a batch of matchups
                src_batch, tgt_batch = src[idxs], tgt[idxs]

                # Randomly shuffle the order for each team to avoid sorted bias
                src_batch_r = src_batch[:, 1:6]
                src_batch_r = src_batch_r[:, torch.randperm(5)]
                src_batch_d = src_batch[:, 7:12]
                src_batch_d = src_batch_d[:, torch.randperm(5)]
                src_batch[:, 1:6] = src_batch_r
                src_batch[:, 7:12] = src_batch_d

                # Randomly shuffle the matchups for half the batch
                is_correct_matchup = np.random.choice([0, 1], batch_size)
                shuffled_lineups = src_batch[is_correct_matchup == 0, 7:12]
                shuffled_lineups = shuffled_lineups[torch.randperm(shuffled_lineups.size()[0])]
                src_batch[is_correct_matchup == 0, 7:12] = shuffled_lineups

                # Generate masks for random heros
                masks = self._gen_random_masks(src_batch, mask_pct)

                src_batch = src_batch.cuda()
                masks = masks.cuda()

                out = self.forward(src_batch, masks)  # -> shape (batch_size, sequence_length, embedding_dim)
                to_predict = out[masks]
                mask_pred = self.get_masked_output(to_predict)
                mask_tgt_batch = tgt_batch[masks].cuda()
                mask_batch_loss = mask_loss(mask_pred, mask_tgt_batch)

                is_correct_pred = self.get_matching_output(out[:, 0, :])
                is_correct_matchup = torch.LongTensor(is_correct_matchup).cuda()
                is_correct_loss = matching_loss(is_correct_pred, is_correct_matchup)
                batch_loss = (mask_batch_loss + is_correct_loss) / 2.
                batch_loss.backward()
                opt.step()

                if step == 0 or (step+1) % print_iter == 0:
                    batch_acc = (mask_pred.detach().cpu().numpy().argmax(1) == mask_tgt_batch.detach().cpu().numpy()).astype(int).mean()
                    top_5_pred = np.argsort(mask_pred.detach().cpu().numpy(), axis=1)[:, -5:]
                    top_5_acc = np.array([t in p for t, p in zip(mask_tgt_batch.detach().cpu().numpy(), top_5_pred)]).astype(int).mean()
                    matching_acc = (is_correct_pred.detach().cpu().numpy().argmax(1) == is_correct_matchup.detach().cpu().numpy()).astype(int).mean()

                    print(f'Step: {step}, Loss: {batch_loss}, Acc: {batch_acc}, Top 5 Acc: {top_5_acc}, Matching Acc: {matching_acc}')
                if (step+1) % save_iter == 0:
                    torch.save(self, f'draft_bert_pretrain_checkpoint_{step}.torch')

    def predict(self, src: torch.LongTensor, mask: torch.BoolTensor, task: DraftBertTasks,
                **predict_kwargs):

        if isinstance(src, (list, np.ndarray)):
            src = torch.LongTensor(src)
        if isinstance(mask, (list, np.ndarray)):
            mask = torch.BoolTensor(mask)
        self.eval()
        if task == DraftBertTasks.DRAFT_PREDICTION:
            src = src.cuda()
            mask = mask.cuda()
            out = self.forward(src, mask)  # -> shape (batch_size, sequence_length, embedding_dim)
            to_predict = out[mask]
            pred = self.get_masked_output(to_predict)

            # pred = self.dense_layer(to_predict)
            return pred