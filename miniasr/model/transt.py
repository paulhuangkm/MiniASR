import torch
import torch.nn as nn
import torch.nn.functional as F
from warprnnt_pytorch import RNNTLoss
from conformer import ConformerBlock
from miniasr.module.masking import truncate_mask


class BaseDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, share_weight=False, tokenizer=None):
        super(BaseDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_proj = nn.Linear(hidden_size, output_size)

        if share_weight:
            self.embedding.weight = self.output_proj.weight

    def forward(self, inputs, length=None, hidden=None):

        embed_inputs = self.embedding(inputs)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths, batch_first=True)

        self.decoder.flatten_parameters()
        outputs, hidden = self.decoder(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        outputs = self.output_proj(outputs)

        return outputs, hidden


def build_decoder(config, tokenizer):
    return BaseDecoder(
        hidden_size=config.dec.hidden_size,
        vocab_size=config.vocab_size,
        output_size=config.dec.output_size,
        n_layers=config.dec.n_layers,
        dropout=config.dropout,
        share_weight=config.share_weight,
        tokenizer=tokenizer
    )

class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, args=None):
        super(BaseEncoder, self).__init__()

        self.pre_linear = nn.Linear(input_size, hidden_size)
        # self.encoder = []
        # for i in range(n_layers):
        #     self.encoder.append(
        #         ConformerBlock(
        #             dim_head = 64,
        #             heads = 2,
        #             **args.enc.conformer
        #         )
        #     )
        # self.encoder = nn.Sequential(*self.encoder)

        self.encoder_layer = nn.TransformerEncoderLayer(
                    **args.enc.transformer
                )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)

        self.output_proj = nn.Linear(hidden_size,
                                     output_size,
                                     bias=True)
        self.n_layers = n_layers

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            # inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)

        # self.encoder.flatten_parameters()
        outputs = self.pre_linear(inputs)
        outputs = self.encoder(outputs, truncate_mask(outputs.shape[0], outputs.shape[0], window_size=200))
        # outputs = self.pre_linear(inputs)
        # for i in range(self.n_layers):
            # outputs = self.encoder_layer(outputs, truncate_mask(outputs.shape[0], outputs.shape[0]))

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            # outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.output_proj(outputs)

        return logits


def build_encoder(config):
    return BaseEncoder(
        input_size=config.feature_dim,
        hidden_size=config.enc.hidden_size,
        output_size=config.enc.output_size,
        n_layers=config.enc.n_layers,
        dropout=config.dropout,
        args = config
    )

class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)

        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs


class Transducer(nn.Module):
    def __init__(self, config, tokenizer):
        super(Transducer, self).__init__()
        # define encoder
        self.config = config
        self.encoder = build_encoder(config)
        # define decoder
        self.decoder = build_decoder(config, tokenizer)
        # define JointNet
        self.joint = JointNet(
            input_size=config.joint.input_size,
            inner_dim=config.joint.inner_size,
            vocab_size=config.vocab_size
        )

        if config.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), '%d != %d' % (self.decoder.embedding.weight.size(1),  self.joint.project_layer.weight.size(1))
            self.joint.project_layer.weight = self.decoder.embedding.weight

        self.crit = RNNTLoss()

    def forward(self, inputs, inputs_length, targets, targets_length):

        enc_state = self.encoder(inputs, inputs_length)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)

        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))

        logits = self.joint(enc_state, dec_state)

        # print(logits.shape)
        # print(targets.shape)
        # print(inputs_length.shape)
        # print(targets_length.shape)
        # print(inputs_length)
        # print(targets_length)
        loss = self.crit(logits.type(torch.float32), targets.int(), inputs_length.int().cuda(), targets_length.int().cuda())
        return loss

    def recognize(self, inputs, inputs_length):

        batch_size = inputs.size(0)

        enc_states = self.encoder(inputs, inputs_length)

        zero_token = torch.LongTensor([[0]])
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []

            dec_state, hidden = self.decoder(zero_token)
            # print(lengths)
            for t in range(lengths):
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                # if t < 10:
                #     print(logits)
                pred = torch.argmax(logits, dim=0)
                pred = int(pred.item())

                if pred != 0:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])

                    if enc_state.is_cuda:
                        token = token.cuda()

                    dec_state, hidden = self.decoder(token, hidden=hidden)
                if pred == 1:
                    token_list.append(pred)
                    break
            # print(token_list)
            return token_list

        results = []
        for i in range(batch_size):
            decoded_seq = decode(enc_states[i], inputs_length[i])
            results.append(decoded_seq)

        return results