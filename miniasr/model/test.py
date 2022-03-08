'''
    File      [ ctc_asr.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ CTC ASR model. ]
'''

import logging
from re import A
import numpy as np
import torch
from torch import nn

from miniasr.model.base_asr import BaseASR
from miniasr.module import RNNEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from warprnnt_pytorch import RNNTLoss
from conformer import ConformerBlock


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
        self.encoder = []
        for i in range(n_layers):
            self.encoder.append(
                ConformerBlock (
                    **args.enc.conformer
                )
            )
        self.encoder = nn.Sequential(*self.encoder)

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
        outputs = self.encoder(outputs)

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
    def __init__(self, enc_size, dec_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.enc_foward = nn.Linear(enc_size, inner_dim, bias=True)
        self.dec_foward = nn.Linear(dec_size, inner_dim, bias=True)

        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=False)

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

        dec_state = self.dec_foward(dec_state)
        enc_state = self.enc_foward(enc_state)
        outputs = enc_state * dec_state

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
            enc_size=config.enc.output_size,
            dec_size=config.dec.output_size,
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

        loss = self.crit(logits.type(torch.float32), targets.int(), inputs_length.int().cuda(), targets_length.int().cuda())
        return loss

    def recognize(self, inputs, inputs_length):

        batch_size = inputs.size(0)

        enc_states = self.encoder(inputs, inputs_length)

        zero_token = torch.LongTensor([[0]])
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):

            dec_state, hidden = self.decoder(zero_token)

            # print(lengths)
            batch_size = enc_state.shape[0]
            token_list = torch.zeros([batch_size, 0])
            t = torch.zeros([batch_size])
            count = 0
            while (t < lengths).any() and count < lengths.max():
                logits = self.joint(enc_state[:, t].view(batch_size, -1), dec_state.view(batch_size, -1))
                pred = torch.argmax(logits, dim=1)

                token_list.append(pred)
                token_list = torch.cat([token_list, pred], dim=1)
                token = torch.LongTensor([[pred]])

                if enc_state.is_cuda:
                    token = token.cuda()

                dec_state, hidden = self.decoder(token, hidden=hidden)
                count += 1
            return token_list

        results = decode(enc_states, inputs_length)

        return results

class ASR(BaseASR):
    '''
        RNNT ASR model
    '''

    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        config = args.model
        config.vocab_size = tokenizer.vocab_size

        self.downsample = nn.Sequential (
            nn.Conv1d(self.in_dim, self.in_dim, 7, 2, 3),
            nn.Conv1d(self.in_dim, self.in_dim, 7, 3, 3)
            # nn.Conv1d(self.in_dim, self.in_dim, 7, 2, 3)
        )
        self.prelinear = nn.Sequential (
            nn.Linear(self.in_dim, self.in_dim * 2),
            nn.ReLU(),

            nn.Linear(self.in_dim * 2, self.in_dim),
            nn.ReLU(),
            nn.Dropout(args.model.dropout)
        )
        self.net = Transducer(config, self.tokenizer)

        # Beam decoding with Flashlight
        self.enable_beam_decode = False
        if self.args.mode in {'dev', 'test'} and self.args.decode.type == 'beam':
            self.enable_beam_decode = True
            self.setup_flashlight()

    def setup_flashlight(self):
        '''
            Setup flashlight for beam decoding.
        '''
        import math
        from flashlight.lib.text.dictionary import (
            Dictionary, load_words, create_word_dict)
        from flashlight.lib.text.decoder import (
            CriterionType, LexiconDecoderOptions, LexiconDecoder, KenLM, Trie, SmearingMode)

        token_dict = Dictionary(self.args.decode.token)
        lexicon = load_words(self.args.decode.lexicon)
        word_dict = create_word_dict(lexicon)

        lm = KenLM(self.args.decode.lm, word_dict)

        sil_idx = token_dict.get_index("|")
        unk_idx = word_dict.get_index("<unk>")

        trie = Trie(token_dict.index_size(), sil_idx)
        start_state = lm.start(False)

        for word, spellings in lexicon.items():
            usr_idx = word_dict.get_index(word)
            _, score = lm.score(start_state, usr_idx)
            for spelling in spellings:
                # convert spelling string into vector of indices
                spelling_idxs = [token_dict.get_index(
                    token) for token in spelling]
                trie.insert(spelling_idxs, usr_idx, score)
        trie.smear(SmearingMode.MAX)

        options = LexiconDecoderOptions(
            self.args.decode.beam_size,
            self.args.decode.token_beam_size,
            self.args.decode.beam_threshold,
            self.args.decode.lm_weight,
            self.args.decode.word_score,
            -math.inf,
            self.args.decode.sil_score,
            self.args.decode.log_add,
            CriterionType.CTC
        )

        blank_idx = token_dict.get_index("#")  # for CTC
        is_token_lm = False  # we use word-level LM
        self.flashlight_decoder = LexiconDecoder(
            options, trie, lm, sil_idx, blank_idx, unk_idx, [], is_token_lm)
        self.token_dict = token_dict

        logging.info(
            f'Beam decoding with beam size {self.args.decode.beam_size}, '
            f'LM weight {self.args.decode.lm_weight}, '
            f'Word score {self.args.decode.word_score}')

    def forward(self, wave, wave_len):
        '''
            Forward function to compute logits.
            Input:
                wave [list]: list of waveform files
                wave_len [long tensor]: waveform lengths
            Output:
                logtis [float tensor]: Batch x Time x Vocabs
                enc_len [long tensor]: encoded length (logits' lengths)
                feat [float tensor]: extracted features
                feat_len [long tensor]: length of extracted features
        '''

        # Extract features
        feat, feat_len = self.extract_features(wave, wave_len)
        
        a = torch.div(feat_len - 1, 2, rounding_mode='floor') + 1
        a = torch.div(a - 1, 3, rounding_mode='floor') + 1
        # a = torch.div(a - 1, 2, rounding_mode='floor') + 1

        return  self.prelinear(self.downsample(feat.transpose(1, 2)).transpose(1, 2)), \
                a, feat, feat_len

    def cal_loss(self, logits, enc_len, feat, feat_len, text, text_len):
        ''' Computes CTC loss. '''

        # log_probs = torch.log_softmax(logits, dim=2)
        # print(feat_len)
        return self.net(logits, enc_len.cpu().int(), text, text_len.cpu().int())
        # Compute loss
        # with torch.backends.cudnn.flags(deterministic=True):
        #     # for reproducibility
        #     ctc_loss = self.ctc_loss(
        #         log_probs.transpose(0, 1),
        #         text, enc_len, text_len)

        # return ctc_loss

    def decode(self, logits, enc_len, decode_type=None):
        ''' Decoding. '''
        if self.enable_beam_decode and decode_type != 'greedy':
            return self.beam_decode(logits, enc_len)
        return self.greedy_decode(logits, enc_len)

    def greedy_decode(self, logits, enc_len):
        ''' CTC greedy decoding. '''
        return [self.tokenizer.decode(h[:enc_len[i]], ignore_repeat=False)
                    for i, h in enumerate(self.net.recognize(logits, enc_len.cpu().int()))]

    def beam_decode(self, logits, enc_len):
        ''' Flashlight beam decoding. '''

        greedy_hyps = self.greedy_decode(logits, enc_len)
        log_probs = torch.log_softmax(logits, dim=2) / np.log(10)

        beam_hyps = []
        for i, log_prob in enumerate(log_probs):
            emissions = log_prob.cpu()
            hyps = self.flashlight_decoder.decode(
                emissions.data_ptr(), enc_len[i], self.vocab_size)

            if len(hyps) > 0 and hyps[0].score < 10000.0:
                hyp = self.tokenizer.decode(hyps[0].tokens, ignore_repeat=True)
                beam_hyps.append(hyp.strip())
            else:
                beam_hyps.append(greedy_hyps[i])

        return beam_hyps
