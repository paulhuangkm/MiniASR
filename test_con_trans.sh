#! /bin/bash

echo "result of val_set:"
miniasr-asr \
    --config ./egs/librispeech/config/transformer_trans_test.yml \
    --test \
    --override "args.data.dev_paths=['data/libri_dev/data_list_sorted.json']" \
    --ckpt model/transformer_trans/epoch=499-step=43499.ckpt

echo "result of test_set:"
miniasr-asr \
    --config ./egs/librispeech/config/transformer_trans_dev.yml \
    --test \
    --override "args.data.dev_paths=['data/libri_test/data_list_sorted.json']" \
    --ckpt model/transformer_trans/epoch=499-step=43499.ckpt