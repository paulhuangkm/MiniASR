'''
    File      [ ctc_asr.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ CTC ASR model. ]
'''

import logging
import numpy as np
import torch
from torch import nn
from conformer import ConformerBlock

from miniasr.model.base_asr import BaseASR
from miniasr.module import RNNEncoder


class ASR(BaseASR):
    '''
        Conformer CTC ASR model
    '''

    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)

        self.prenet = nn.Conv1d(self.in_dim, args.model.encoder.dim, 7, 2, 3)

        self.encoder = []
        for i in range(4):
            self.encoder.append(
                ConformerBlock(
                    dim_head = 64,
                    heads = 2,
                    **args.model.encoder
                )
            )
        self.encoder = nn.Sequential(*self.encoder)
        self.ctc_output_layer = nn.Linear(
            args.model.encoder.dim, self.vocab_size)
        
        # Loss function (CTC loss)
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', blank=0, zero_infinity=True)
        self.kldivloss = torch.nn.KLDivLoss(reduction='batchmean')
        self.weight = 0.05

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

        # Encode features
        feat = feat.transpose(1, 2)
        feat = self.prenet(feat)
        feat = feat.transpose(1, 2)
        
        enc, enc_len = self.encoder(feat), feat_len // 2

        # Project hidden features to vocabularies
        logits = self.ctc_output_layer(enc)

        return logits, enc_len, feat, feat_len

    def cal_loss(self, logits, enc_len, feat, feat_len, text, text_len):
        ''' Computes CTC loss. '''

        log_probs = torch.log_softmax(logits, dim=2)
        kl_inp = log_probs.transpose(0, 1)
        kl_tar = torch.full_like(kl_inp, 0.01)
        kldiv_loss = self.kldivloss(kl_inp, kl_tar)

        # Compute loss
        with torch.backends.cudnn.flags(deterministic=True):
            # for reproducibility
            ctc_loss = self.ctc_loss(
                log_probs.transpose(0, 1),
                text, enc_len, text_len)

        loss = (1. - self.weight) * ctc_loss + self.weight * kldiv_loss
        return loss / self.args.hparam.val_batch_size

    def decode(self, logits, enc_len, decode_type=None):
        ''' Decoding. '''
        if self.enable_beam_decode and decode_type != 'greedy':
            return self.beam_decode(logits, enc_len)
        return self.greedy_decode(logits, enc_len)

    def greedy_decode(self, logits, enc_len):
        ''' CTC greedy decoding. '''
        hyps = torch.argmax(logits, dim=2).cpu().tolist()  # Batch x Time
        return [self.tokenizer.decode(h[:enc_len[i]], ignore_repeat=True)
                for i, h in enumerate(hyps)]

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
