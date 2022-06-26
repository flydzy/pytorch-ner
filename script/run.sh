# define for data path
DATA_PATH=./data
LANG_NAME=(en de es nl)


for lang in ${LANG_NAME[@]};do
    echo "Processing ${lang}"
    echo ${DATA_PATH}/${lang}/train.txt
    echo ${DATA_PATH}/${lang}/test.txt
    echo ${DATA_PATH}/${lang}/dev.txt
    python3 main.py \
        --train_path ${DATA_PATH}/${lang}/train.txt \
        --dev_path ${DATA_PATH}/${lang}/dev.txt \
        --test_path ${DATA_PATH}/${lang}/test.txt \
        --lang ${lang} 
done 
