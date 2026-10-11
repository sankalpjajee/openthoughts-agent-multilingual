# OpenThoughts-Agent Multilingual

Multilingual extension of [OpenThoughts-Agent](https://github.com/sankalpjajee/OpenThoughts-Agent)
(OT-Agent): converting its agentic data, training, and evaluation pipelines into languages other than
English, building on the IndicMMLU-Pro work.

## Status

**Skeleton only.** This repository currently holds the empty directory structure of OT-Agent (each
directory carries a `.gitkeep`), plus the upstream `LICENSE.txt` and `.gitignore`. No code has been
ported yet.

Structure mirrored from `sankalpjajee/OpenThoughts-Agent@3bd1917e62c9d03d73063b433f5c442c279c0563`.

## Open decisions

The codebase survey in [docs/OT_AGENT_MULTILINGUAL_SURVEY.md](docs/OT_AGENT_MULTILINGUAL_SURVEY.md) maps what
OT-Agent consists of, where natural language lives, the candidate workstreams, a recommended pilot, and a full
decision table.

- [ ] **What to convert** — task datasets (`data/`), agent prompts / chat templates, eval benchmarks
      (`eval/`), generated traces, or a combination.
- [ ] **Target languages** — e.g. the Indic languages covered by IndicMMLU-Pro, optionally with
      high-resource controls.
- [ ] **Relationship to upstream** — standalone port vs. thin layer that tracks OT-Agent and adds
      translations on top.

## Layout

Mirrors OT-Agent to two levels for infrastructure directories. Roles below describe the upstream
counterpart.

| Directory | Upstream role |
|---|---|
| `data/` | Data pipelines: Harbor task datasets and agent traces, one subdirectory per pipeline |
| `hpc/` | Unified launcher (`python -m hpc.launch --job_type <type>`), cluster sbatch / YAML configs |
| `eval/` | Listener-driven eval pipeline: serve with vLLM, run trials via Harbor + Daytona |
| `rl/` | RL training (SkyRL / GRPO) per cluster |
| `sft/` | SFT configs (LLaMA-Factory, axolotl) |
| `scripts/` | Utilities: analysis, database, datagen, Daytona, Harbor, Hugging Face, W&B |
| `tests/` | Test suite |
| `model_config/` | Per-model vLLM serve configs |
| `baselines/` | Baseline reproductions |
| `prm/` | Trajectory PRM judges (e.g. thrashing detection) |
| `chat_templates/` | Jinja chat templates |
| `database/`, `schema/` | Dataset / model / eval registry and its schema |
| `docker/` | Container images |
| `docs/` | Reference documentation |
| `notebook/` | Datagen + SFT tutorial notebook |
| `assets/` | Static assets |
| `.github/workflows/` | CI |

### Not mirrored

- Per-item subdirectories of `data/`, `model_config/`, and `baselines/` — added once we pick what to
  convert.
- `sft/llamafactory`, `sft/axolotl` — git submodules upstream; add as submodules if needed.
- `scripts/_deprecated/`, and operator- / cluster-specific `.agents/` and `.kube/`.

## License

Apache License 2.0, as upstream. See [LICENSE.txt](LICENSE.txt).
