# Before/after evaluation

Base model: `Qwen/Qwen2.5-0.5B-Instruct` | adapter: `models/adapter`

Test set: 1000 held-out synthetic examples (data\test.jsonl), greedy decoding, max_new_tokens=64, device: cuda. Evaluated on 2026-07-07.

| Metric | Base model (zero-shot) | LoRA fine-tuned | Delta |
|---|---|---|---|
| JSON validity | 100.0% | 100.0% | +0.0 pp |
| Category accuracy | 18.6% | 98.2% | +79.6 pp |
| Merchant name exact match | 16.9% | 97.2% | +80.3 pp |
| Both fields correct | 4.3% | 97.2% | +92.9 pp |

Malformed or missing JSON counts as wrong on every metric, for both models.
