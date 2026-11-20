# Sphere benchmark (Windows)

From the WISE checkout, with Python 3.11 and FFmpeg 6 available:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/eval/sphere_benchmark.py --sphere-eval C:\Users\luoyw\work\lab\Sphere\sphere-eval --archives D:\work\Sphere\benchmark_videos --ffmpeg .runtime/ffmpeg-6.1.1-full_build-shared/bin/ffmpeg.exe --output sphere-results.md
```

Append `--align-only` and choose a different output path to validate the archives without downloading or loading the model. The normal invocation downloads the default WISE visual model and its tokenizer on first use. The report is replaced at startup and updated after each dataset; a process error returns a nonzero exit status. SHA-256 mismatches are reported as skips. The runner never reads or removes existing WISE projects and uses disposable per-dataset storage and search indexes. The report describes the offline visual baseline; it is not a score for WISE's full web search pipeline.

## Model cache used for the recorded run

- Repository: `timm/ViT-B-16-SigLIP2-512`
- Revision: `5e7f4f980e465036486ce78dfc721bbedf291939`
- `open_clip_model.safetensors` SHA-256: `c96b5b6ed1dee282ad4a4665caa1bd74cda97f46e1e3e80e1c022752bf054927`

With this revision and tokenizer files available in the local Hugging Face cache, set `$env:HF_HUB_OFFLINE='1'` and `$env:TRANSFORMERS_OFFLINE='1'` to avoid network requests while benchmarking.
