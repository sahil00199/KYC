python train_vanilla.py --data_dir ./data/processed \
--model_type bert \
--labels data/labels.txt \
--model_name_or_path bert-base-cased \
--output_dir models/vanilla/ \
--max_seq_length  128 \
--num_train_epochs 3 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 64 \
--num_train_epochs 3 \
--save_steps -1 \
--seed 7 \
--do_eval  #--do_train

