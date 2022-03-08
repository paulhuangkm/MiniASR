echo "result of val_set:"
miniasr-asr \
    --config ./egs/librispeech/config/con_ctc_adaptive_dev.yaml \
    --test \
    --override "args.data.dev_paths=['data/libri_dev/data_list_sorted.json']" \
    --ckpt model/con_libri-10h_char_adaptive/epoch=479-step=41759.ckpt
echo "result of test_set:"
miniasr-asr \
    --config ./egs/librispeech/config/con_ctc_adaptive_test.yaml \
    --test \
    --override "args.data.dev_paths=['data/libri_test/data_list_sorted.json']" \
    --ckpt model/con_libri-10h_char_adaptive/epoch=479-step=41759.ckpt