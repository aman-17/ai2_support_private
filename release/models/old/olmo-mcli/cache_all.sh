
seq 1 557| parallel -j 10 --bar --max-args=1 python cache_all.py s3://ai2-llm/checkpoints/7b/v1_5-mix-mitch-ish/step{1}000
