# Cổng chất lượng: results/gd1/09-tang-cho-kv/20261014-p09/quality-kv8

So với baseline lần 1 `results/gd1/06-chat-luong-baseline/20261013-p06` và lần 2 `results/gd1/06-chat-luong-baseline/20261013-p06r2`.

- **finqa**: baseline 59.2 / 60.1, biến thể 59.0; hiệu so lần 1 -0.17 (CI -1.78 … +1.43), so lần 2 -1.05; flips 7.7% / 7.7% (tự nhiên 4.5%)
- **mmlu_pro**: baseline 69.0 / 67.4, biến thể 67.8; hiệu so lần 1 -1.20 (CI -3.38 … +0.98), so lần 2 +0.40; flips 12.4% / 11.0% (tự nhiên 6.6%)
- **sinh tự do**: giống hệt 5.4% (baseline tự lệch 23.1%), vị trí lệch p50 19.0 (baseline 46.0); Δp p1/p99 -0.163 / +0.171; KL TB 0.0113 (baseline 0.0015)
- **teacher forcing**: same top-1 97.09% (baseline c1 vs c8 99.10%); Δp p1/p99 -0.170 / +0.131; KL TB 0.0100

**Kết luận:** dấu hiệu kém đi → chạy lại cổng một lần
- finqa: flips 7.7% > 1.5 × flips tự nhiên 4.5%
- mmlu_pro: flips 12.4% > 1.5 × flips tự nhiên 6.6%
