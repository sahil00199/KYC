python train_kyc.py \
--do_lower_case \
--data_dir ./data/processed \
--bert_type bert-base-uncased \
--max_seq_length 256 \
--train_batch_size 16 \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--output_dir ./models/kyc \
--verbosity 1 \
--seed 7 --do_train

python train_vanilla.py \
--do_lower_case \
--data_dir ./data/processed \
--bert_type bert-base-uncased \
--max_seq_length 256 \
--train_batch_size 16 \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--output_dir ./models/vanilla \
--verbosity 1 \
--seed 7 --do_train


python get_scores.py
