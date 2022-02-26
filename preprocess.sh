# Train set
miniasr-preprocess \
        -c LibriSpeech \
        -p data/librispeech_finetuning \
        -s 1h \
        -o data/libri_train_1h \
        --gen-vocab \
        --char-vocab-size 40

miniasr-preprocess \
        -c LibriSpeech \
        -p data/librispeech_finetuning \
        -s 9h \
        -o data/libri_train_9h

# Development set
miniasr-preprocess \
        -c LibriSpeech \
        -p data/LibriSpeech \
        -s dev-clean \
        -o data/libri_dev

# Test set
miniasr-preprocess \
        -c LibriSpeech \
        -p data/LibriSpeech \
        -s test-clean \
        -o data/libri_test