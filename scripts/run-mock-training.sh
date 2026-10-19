uv run torchrun --standalone --nproc_per_node=2 --local-addr 127.0.0.1 \
  -m vrollout.train \
  --backend mock \
  --group-size 4 \
  --num-iterations 3
