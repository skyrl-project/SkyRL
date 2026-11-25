# Tmax through skycap

[Tmax](https://arxiv.org/abs/2606.23321) (Ivison et al.) is an open RL recipe for terminal agents: 13,565
generated terminal tasks, each a Docker image with a programmatic verifier, and a small bash-tool harness,
Vanillux2. This recipe trains on those tasks through [skycap](../README.md), with Tmax's own harness, and
evaluates on TB-Lite as Tmax does.

```bash
SANDBOX_OWNER=<you> DAYTONA_KEY_FILE=<file with DAYTONA_API_KEY=...> \
  bash examples/train_integrations/harbor_skycap/tmax/run_tmax.sh
```

Defaults: Qwen3.5-2B, LoRA r32, 2 GPUs colocated, 8 tasks × 8 samples per step, 16 steps, 64k context, TB-Lite ×
1 before the first step and after the last. `run_tmax.sh` lists every knob. It is Megatron-only: Qwen3.5's
248k-token vocabulary needs `trainer.fused_lm_head_logprob=true`, and FP8 rollouts need serialized FP8 weight
sync, both of which only the Megatron backend has.

## The tasks

`prepare_tmax.py` (run by the script, once) writes two roots under `DATA_ROOT` (default `~/data/tmax`):

- `tmax/`: Tmax v1.1's Harbor export (`TMaxxx/tmax-13k-v1.1-harbor`, pinned revision, checked against its sha256).
  Training draws from all of it, as Tmax does.
- `tblite/`: [OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite)'s 100 tasks
  (`openthoughts-tblite@2.0` in Harbor's registry), the set Tmax picks checkpoints on.

Every Tmax task ships an `environment/docker-compose.yaml` that only pins the platform, which its Dockerfile
already does. Harbor's Daytona environment runs any task with a compose file in compose mode, a Docker-in-Docker
sandbox that runs `docker compose build` on every trial, so `prepare_tmax.py` deletes compose files that do
nothing else. Each sandbox then builds its task's Dockerfile. With `AUTO_SNAPSHOT=1`, Harbor instead builds one
Daytona snapshot per task environment and starts every sandbox of that task from it, so each task image is pulled
from Docker Hub once; that needs a Daytona key allowed to create snapshots.

The sandboxes get each task's own resources and timeouts (Tmax: 1 CPU, 2 GB, 600 s for the agent, 300 s for the
verifier): the script clears the 1 CPU / 1 GB / 1200 s overrides of the shared Harbor defaults.

## The harness

`Vanillux2Agent` (from [hamishivi/tmax](https://github.com/hamishivi/tmax), pinned with `TMAX_REF`) is a
driver-side Harbor agent, like Terminus-2: it calls the model from the generator's process with native OpenAI
tool calls and runs each command in the sandbox with `environment.exec`. So it needs no `skycap.exposure` and
installs nothing into the task's system. It reads parsed `tool_calls`, so the recipe sets
`skycap.use_raw_content=false`: skycap parses each reply with the model's renderer instead of answering with raw
text. Vanillux2 sends each parsed reply back as it got it, reasoning included, so a thinking model's history stays
the conversation it sampled; that needs LiteLLM 1.110 or newer (the script installs it), since older versions drop
`reasoning_content` from what they send. When a prompt fills the context, skycap answers `context_length_exceeded`, which LiteLLM raises as a
`BadRequestError`: Vanillux2 retries it five times (about 30 s), then stops, and the trial trains with reward 0.

The recipe trains synchronously only. Vanillux2 takes no `llm_kwargs`, so the generator's prefix-cache salt
never reaches skycap. That is a no-op here, because each synchronous weight sync resets vLLM's prefix cache,
but it would not be under fully-async training.

## Algorithm

| Setting | Value | From |
| --- | --- | --- |
| Loss | DPPO, binary TV, δ 0.15 / 0.15 | SkyRL's ApexAgents recipe; Tmax uses 0.1 |
| Advantage, aggregation | GRPO without std normalization, `prompt_mean` | SkyRL defaults |
| Zero-variance groups | masked (`zero_variance_filter=true`) | Tmax |
| Sampling | temperature 1.0, top-p 1.0, top-k 64 | Tmax; top-k because score centering needs it |
| Context | 64k total, 16k per reply, 64 model calls per trial | Tmax |

`FP8=1` sends FP8 rollout weights (`fp8_weight_sync_mode=blockwise`; LoRA stays merged so every sync carries
full weights). `CENTER=1` turns on sample-support replay and score centering. The two are independent switches,
so the four combinations can be compared on the same tasks (same `SEED`); see the
[off-policy correction guide](../../../../docs/content/docs/algorithms/off_policy_correction.mdx) for the metrics.

## Models

skycap picks its renderer by exact model name, from the `renderers` package. Qwen3.5-0.8B through 9B have one;
`Qwen/Qwen3.6-27B`, Tmax's largest model, does not yet.
