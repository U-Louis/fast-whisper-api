# fast-whisper-api

[Faster Whisper](https://github.com/SYSTRAN/faster-whisper) behind the same HTTP contract as whisper-api.

`POST /v1/transcribe` returns `{"text","language"}`. The only request field besides the file is an optional two-letter language. The model stays `small`. The task stays transcribe.

`WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, and `WHISPER_CPU_THREADS` are read from the environment. CPU uses `int8`. CUDA uses `float16`. Beam size is 1. A request cannot set any of these.

`Dockerfile` is the CPU image. `Dockerfile.cuda` is the GPU override. Both bake the `small` weights during the image build.
