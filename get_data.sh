mkdir -p data
cd data
wget https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz
tar zxf librispeech_finetuning.tgz
rm librispeech_finetuning.tgz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar zxf dev-clean.tar.gz
tar zxf test-clean.tar.gz
rm dev-clean.tar.gz
rm test-clean.tar.gz
cd ..