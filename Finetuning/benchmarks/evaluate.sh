# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export TOKENIZERS_PARALLELISM=0

python benchamrks/benchmarks.py \
--dataset finred \
--base_model chatglm2 \
--peft_model ../finetuned_models/headline-chatglm2-linear_202311221016 \
--batch_size 8 \
--max_length 512
