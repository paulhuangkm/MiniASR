echo "result of val_set:"
miniasr-asr \
    --config ./egs/librispeech/config/las_testing.yaml \
    --test \
    --override "args.data.dev_paths=['data/libri_dev/data_list_sorted.json']" \
    --ckpt model/las/epoch=289-step=99999.ckpt
echo "result of test_set:"
miniasr-asr \
    --config ./egs/librispeech/config/las_testing.yaml \
    --test \
    --override "args.data.dev_paths=['data/libri_test/data_list_sorted.json']" \
    --ckpt model/las/epoch=289-step=99999.ckpt