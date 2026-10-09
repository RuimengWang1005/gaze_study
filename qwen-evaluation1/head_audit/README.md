# ViT head audit

This package is an audit-only path for capturing per-head Qwen2.5-VL vision
attention and checking its spatial mapping. It does not change the GAZE
selector or train any model. Generated artifacts belong under
`results/vit_head_audit_v1/` and are intentionally ignored by Git.

Run CPU tests with the project environment:

```bash
conda run -n vs_env pytest -q qwen-evaluation1/tests/test_head_attention_capture.py qwen-evaluation1/tests/test_head_token_mapping.py
```

Run the one-image smoke audit before expanding to the small mapping audit:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n vs_env python qwen-evaluation1/audit_vit_heads.py --num-samples 1
```

No head-selection output is produced by this code.
