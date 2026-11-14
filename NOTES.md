# NOTES

Running log of where the official docs differ from the build spec, plus decisions and open items.

## M0: docs check (2026-11-12)

Sources read: `docs.typesafe.ai` (`/api`, `/models`, `/api/errors`, `/primitives/noul`, `/concepts/state`,
`/patterns/probability-routing`, `/changelog`, `/agent-skill`, Python SDK reference pages), PyPI `typesafe-sdk`,
OpenRouter's Jev guide and the Haiku 5.5 / Opus 5.5 model pages.

### Confirmed as written in spec section 3

- Endpoint `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`.
- `GET /v1/models` lists models and sends no text (SDK: `client.models.list()`).
- Body is `model`, `state`, `questions`; response is `model`, `answers`, `usage.input_tokens`, `usage.output_tokens`.
- `noul` returns one probability from 0 to 1 (probability of yes/true). No separate confidence field.
- Price: $0.042 per million input tokens, output free. Same for 1.13 and 1.14.
- SDK: `typesafe-sdk` (latest 0.7.5, Python >= 3.10). `TypeSafeClient()` reads `TYPESAFE_API_KEY`.
  Result access `response.nouls["small_ok"].noul` is correct. Also `response.model`, `response.usage`, `response.request_id`.
- 401 / 422 / 429 / 529 mean what the spec says. 429 and 529 are the retryable ones.

### Differences from the spec

| # | Spec said | Docs say | Effect |
|---|---|---|---|
| 1 | Pin `jev-1.13.0` "or whatever is current" | `jev-1.14.0` was released 2026-10-14. `jev-latest` and `jev-preview` both point to it. `jev-1.13.0` is still served. No deprecation dates published. | **Design choice, needs OK.** Proposal: pin `jev-1.14.0`. |
| 2 | Backticks in `instructions` reference state fields | Not documented. The docs describe no field-reference syntax. They recommend an object state with descriptive field names and let the model infer. One example uses backticks only as a naming convention. | Keep `state={"prompt": ...}`. Do not rely on backticks doing anything. Phrase the question in plain words ("the request in the prompt field"). A backtick variant can be one of the 3 dev phrasings. |
| 3 | Limit: plan around 32k tokens of state | 32k tokens per question (state + one question), 64k per request (state + all questions). 262k with the `TypeSafe-Danger-Zone-Long-Context: true` header, which is slower and less accurate. | None. Our prompts are a few hundred tokens. |
| 4 | Three question types | Four: `choice`, `score`, `noul`, `box` (image bounding box, 1.14 only). | None. |
| 5 | `score` uses 2 to 10 rubric levels | `criteria` is an array of 1 to 10 items. Levels are zero-based by position. `score` is a probability-weighted average, so it can be fractional. | Stretch goal only: a 5-level rubric scores 0 to 4. |
| 6 | Errors: 401, 422, 429, 529 | Also 400 (rule broken, or `max_tokens_exceeded` context overflow), 413 (body over 32 MiB), 415 (bad compression). | Treat 400/401/413/415/422 as non-retryable. |
| 7 | "Retry 429 and 529 with exponential backoff" (implies we write it) | The SDK already retries: `RetryPolicy` defaults are 2 retries, 0.5 s initial backoff doubling to 5 s max, statuses 408/429/5xx, honours `Retry-After`, 30 s total budget per call. | Do not hand-roll retries. Eval: `RetryPolicy(max_retries=5)` or so. **Live router: `RetryPolicy(max_retries=0, timeout=2.0)`** so the 2 second fail-safe actually fires instead of the SDK retrying for up to 30 s. |
| 8 | SDK sample omits `model` | `system_one(state, questions, model=None, ...)`. `None` falls back to the client default or `TYPESAFE_DEFAULT_MODEL`. | Always pass `model=` from `config.yaml`. Log `response.model` per call as planned. |
| 9 | `noul` takes `instructions` | Also takes optional `criteria` (`NoulCriteria(true=..., false=...)`) to define the yes/no boundary. | Candidate for one of the 3 dev phrasings. |
| 10 | OpenRouter: `typesafe/jev-1.13`, "System One-compatible endpoint under https://openrouter.ai/api" | Endpoint is `https://openrouter.ai/api/v1/systemone`. IDs: `typesafe/jev-1.14` (current), `typesafe/jev-1.13`, alias `~typesafe/jev-latest`. The TypeSafe SDK works with a base URL change (`base_url=` or `TYPESAFE_BASE_URL`) and an OpenRouter key. Responses add `usage.cost`. Requests over about 32,000 bytes of text run in long-context mode. | Viable fallback if there is no TypeSafe key. One key could then cover everything. |
| 11 | Skill repo is optional | Install is `claude plugin marketplace add typesafe-ai/skills` then `claude plugin install typesafe@typesafe-ai`. | Not installed. Will ask before doing so. |

### Guidance from the docs worth using

- Phrase a noul so a high value means yes. Ask one condition per noul. Both hold for the starting question.
- "A statement can work instead of a question. Test both phrasings on your data." This maps onto the
  3-phrasing budget: (a) the spec's statement, (b) a question form, (c) statement plus `criteria`.
- Questions in one request are evaluated in parallel and in isolation. All 3 phrasings can therefore go in
  **one** Jev call per dev prompt at close to the latency and cost of one. Latency for the report must still
  be measured with the single final question.
- The docs suggest 0.7 / 0.3 as starting thresholds and warn against tuning on a handful of cases. We sweep on dev anyway.

### Not verified

- Vercel AI Gateway (`typesafe-ai/jev`, noul called `boolean`): not checked. Not needed unless OpenRouter is ruled out.
- OpenRouter's Jev price: the model page did not load for me. Their guide says input-billed, output free. Treat $0.042/M as likely but unconfirmed there; `usage.cost` will settle it.
- `typesafe/jev-router` model ID: the guide describes a "Jev Router" but gives no ID. Stretch only.
- Whether text inside `state` can act as instructions (prompt injection). Docs are silent. Benchmark prompts make this low risk here; it belongs in README limitations for real traffic.
- Jev latency: no published figures. We measure p50/p95 ourselves.

## Decisions after M0 (2026-11-12)

- **LLMs run locally on Ollama, not OpenRouter** (user decision). This replaces the Haiku/Opus proposal below.
  - small: `qwen3:1.7b` (1.4 GB, not pulled yet). large: `qwen3:8b` (already installed).
  - Hardware is an RTX 4060 Laptop GPU with 8 GB VRAM, which is why the large model stops at 8B.
  - Real spend is $0 for the LLMs. Only Jev is billed (about a cent for the whole project).
  - **The cost axis uses imputed prices.** `qwen3:8b` is priced at OpenRouter's `qwen/qwen3-8b` list rate,
    $0.117 / $0.455 per million input / output tokens (read 2026-11-12). No hosted price was found for the
    1.7B model, so it is scaled by parameter count (1.7 / 8): $0.025 / $0.097. That scaling is an assumption,
    not a quote. The price ratio is about 4.7x, far smaller than Haiku/Opus at 40x, so percent-saved figures
    will be modest by construction. The README must say this.
  - Ollama is called through its native `/api/chat` with `think: false`. Qwen3 is a thinking model and its
    reasoning tokens would otherwise eat the 600-token output cap. Checked locally on `qwen3:8b`.
  - Local latency is not representative of hosted models. Jev latency (the reported one) is unaffected.
- **Jev pinned to `jev-1.14.0`** (current release; the spec allowed "whatever is current").
- **Jev goes direct to TypeSafe** with `TYPESAFE_API_KEY` from `.env`.
- The OpenRouter client is kept behind the same interface (`provider: openrouter` in `config.yaml`) so the
  pair can be swapped later. It is covered by a mocked test only; it has never been run against OpenRouter.
- The live router (`router/route.py`, spec section 9) was built in M1 because the fallback and threshold tests
  need it. The deadline is enforced with a worker thread on top of the SDK timeout, and the real client is
  created with `max_retries=0`.
- Tooling: `uv` is not installed, so the project uses `python -m venv .venv` plus pip. Python is 3.13.

## M2: dataset and graders (2026-11-12)

- All four suggested sources load: `cais/mmlu` (config `all`), `allenai/ai2_arc` (`ARC-Challenge`),
  `openai/gsm8k` (`main`), `TIGER-Lab/MMLU-Pro`. The `test` split of each is used.
- `data/prompts.jsonl` has 400 prompts, 100 per source, 50 dev and 50 test per source. It carries a `split`
  field in addition to the spec's `id`, `source`, `prompt`, `gold`. Rebuilding gives a byte-identical file.
- Grading is strict: only the last `ANSWER:` line counts. An output with no such line, including one cut off
  at the 600-token cap, is graded wrong. This can understate a model that reasons at length.
- ARC rows labelled 1-4 are re-lettered A-D. MMLU-Pro `N/A` options are dropped and the rest re-lettered.
- MMLU-Pro has up to 10 options (A-J), so chance accuracy there is about 10%, against 25% for MMLU and ARC.
- MMLU-Pro draws part of its questions from MMLU. The current sample has no duplicate prompts.

## M3: smoke test (2026-11-12)

`make smoke ARGS=--yes` ran 10 dev prompts end to end. Real spend $0.000176.

- **`GET /v1/models` lists only the aliases** (`jev-latest`, `jev-preview`), not pinned versions. A request
  with `model: jev-1.14.0` is accepted anyway and every response reported `jev-1.14.0`. The smoke script
  therefore treats "not in the list" as a note, not an error.
- **Jev bills about 300 tokens of fixed overhead per question** on top of the state (335 input tokens for a
  139-character prompt). Output is reported as 21 tokens and is free. My first estimate assumed 60 and came
  out low; the estimator now uses 350. The full run is still under one cent ($0.0076 for 400 prompts).
- Jev latency in this run: 56 to 127 ms per call, single question. Too few calls for p50/p95.
- 10-prompt accuracy, for plumbing only: small 4/10, large 8/10. Not a result.
- One small-model reply ended "so the correct answer is D." with no `ANSWER:` line and was graded wrong
  although D was the gold answer. Strict grading stands for now; the M4 pilot will count how often this
  happens for each model, because a format failure that hits the small model more often would inflate the gap.
- One hard MMLU-Pro chemistry item ran both models into the 600-token cap with no answer.
- Windows consoles default to cp1252 and cannot print some model output. Scripts now switch stdout to UTF-8.

## M4: pilot and gap check (2026-11-13)

`make pilot ARGS=--yes` ran 50 dev prompts into the real caches (`cache/llm.jsonl`, `cache/jev.jsonl`).
Real spend $0.000892. The full run will resume from these rows.

| source | n | small (`qwen3:1.7b`) | large (`qwen3:8b`) | gap |
|---|---|---|---|---|
| arc | 13 | 0.615 | 0.923 | +30.8 |
| gsm8k | 12 | 0.750 | 0.917 | +16.7 |
| mmlu | 13 | 0.692 | 0.769 | +7.7 |
| mmlu_pro | 12 | 0.417 | 0.583 | +16.7 |
| all | 50 | 0.620 | 0.800 | **+18.0** |

- **Gap check passes**: 18 points against the 8-point minimum. No change of sources or models is proposed.
  With 12 or 13 prompts per source the per-source figures are noise; only the overall gap is used.
- Outcome mix: both right 29, only large 11, only small 2, neither 8. Oracle accuracy 0.84. Routing to large
  only helps on 22% of prompts, which is the headroom any router has.
- **Format failures**: 2 replies, both from the small model, gave the right letter without an `ANSWER:` line
  ("so the correct answer is D." and "Final line: **D**"). Under lenient grading small would be 0.66 and the
  gap 14 points. Grading stays strict: the instruction is part of the task, and a production caller parsing
  the output would also fail on these. The README should state the rate.
- Token cap: 3 replies per model hit 600 tokens with no answer, all counted wrong.
- **Local runs are not perfectly repeatable.** `mmlu-10082` on the small model answered C in the smoke run and
  D in the pilot, at temperature 0 with the same seed. The caches are the record; a fresh run may differ slightly.
- Preview only, 50 dev prompts, too few to conclude anything: AUROC for "small is correct" was 0.651 for Jev
  and 0.613 for prompt length. Jev probabilities ranged 0.25 to 0.96. Jev latency p50 78 ms, max 388 ms.

## M5: full eval, simulation, chart (2026-11-13)

Total real spend for M5: about $0.011, all Jev (phrasings $0.0045, full run $0.0061). Numbers below are
from `results/metrics.json`; rebuild with `make results` (no API calls).

### Phrasing choice (dev only)

Three phrasings were asked in one request per dev prompt (`cache/jev_phrasings_dev.jsonl`). Three questions
cost about 560 input tokens against 340 for one, so the state is billed once, not per question.

| phrasing | dev AUROC for "small is correct" | 95% CI |
|---|---|---|
| statement (spec's wording) | 0.718 | 0.646 to 0.787 |
| question form | 0.712 | 0.638 to 0.783 |
| question plus `criteria` | 0.657 | 0.580 to 0.738 |
| prompt length (baseline) | 0.668 | 0.591 to 0.747 |

`statement` was kept. It and the question form are indistinguishable; adding `criteria` made it worse.
The phrasing budget of 3 is now used up.

### Result (test split, 200 prompts)

- Always-small 58.0% at $0.0149 per 1,000 queries. Always-large 80.0% at $0.0760. Oracle 83.5% at $0.0308.
- **Headline operating point**: the dev-chosen Jev threshold is 0.90, which routes 92% of test prompts to
  large. Accuracy 78.5% (1.5 points below always-large, CI -3.5 to 0.0) at $0.0914, which is
  **20.3% more expensive than always-large** (CI 18.2% to 22.5% more). There is no saving.
- The prompt-length baseline, tuned the same way, sends 99% to large: same accuracy, 0.6% cheaper.
- **Jev does carry signal about the small model**: AUROC for "small is correct" is 0.725 (CI 0.655 to 0.798)
  against 0.589 for prompt length; the difference of 0.137 has a CI of 0.052 to 0.221, which excludes zero.
- **That signal does not help routing here.** AUROC for "large is needed" (small wrong, large right) is
  0.495 for Jev and 0.444 for length: chance. At a half-and-half split Jev is 0.8 points below random
  routing at the same share (CI -4.2 to +2.6).
- Jev latency over 400 single-question calls: p50 71 ms, p95 119 ms.

### Why the curve is below the random line

| Jev p bin (test) | n | small acc | large acc | mean large cost per 1,000 |
|---|---|---|---|---|
| 0.18 to 0.55 | 40 | 32.5% | 32.5% | $0.149 |
| 0.56 to 0.70 | 40 | 35.0% | 75.0% | $0.095 |
| 0.70 to 0.79 | 40 | 67.5% | 95.0% | $0.056 |
| 0.79 to 0.85 | 40 | 77.5% | 97.5% | $0.043 |
| 0.85 to 0.95 | 40 | 77.5% | 100% | $0.038 |

1. The prompts Jev rates hardest (mostly MMLU-Pro) are ones the large model also gets wrong, and they are the
   most expensive to send to it (about twice the mean). A threshold on `p_small_ok` sends these first.
   On dev the bottom bin did gain 17 points, so this is partly a test-split draw, but the pattern is the same.
2. The large model's gain is spread fairly evenly over the other bins, including 22 points in the top bin,
   where the small model is "probably right" but the large one is almost always right.
3. Jev's own cost ($0.0176 per 1,000 queries) is more than the small model's ($0.0149) and 23% of the large
   model's at these imputed prices. With a 5x price ratio the routing fee eats most of the possible saving.
   With a hosted pair at a 40x ratio the same fee would be negligible. That is a statement about arithmetic,
   not a measured result.

"Will the small model be right?" is the question the spec asks, and Jev answers it moderately well. The
routing decision needs a different one: "will the large model be right where the small one is not?".

### Things that went wrong during the run

- Ollama stalled on one ordinary request (GPU idle, no response) and later returned a 500. The client now
  times out after 180 s and retries stalls and 5xx responses twice.
- One large-model reply (`mmlu_pro-3941`) hit Ollama's "token repeat limit reached" abort on every attempt:
  the model loops on one token. It is recorded as an empty, wrong answer with an `error` field, and costed
  as a run to the 600-token cap, which is what a hosted API would have billed.
- Format failures over all 400 prompts: small 23 replies with no `ANSWER:` line (7 at the token cap),
  large 14 (all at the cap). So 16 small-model replies (4%) lost on format alone, against 0 for large.
  Strict grading therefore widens the gap by up to 4 points; the full-run gap is 22.8.

## M5b: cascade (2026-11-13)

Added after the pre-routing result above was known. Real spend about $0.015, all Jev.

### What changed in the repo

- `router/` now holds only the live router (config, costs, Jev and LLM clients, routing). Everything used
  only for evaluation moved to `evalkit/` (cache, cli, dataset, graders, runs, simulate). `eval/` still holds
  the scripts. No behaviour change: `results/metrics.json` was byte-identical before and after the move.
- New: `eval/run_checks.py`, `eval/pick_check.py`, `Cascade` in `router/route.py`, `python -m router --cascade`.

### The pattern

TypeSafe's own cookbook (`docs.typesafe.ai/cookbooks/sde_cascade`) reports a cascade: a cheap model answers,
Jev checks the answer, a strong model answers only when a check fires. Their numbers: cheap model 92.6% at
$0.156 per 1,000 tasks, strong model 98.1% at $1.89, cascade 96.5% at $0.49 with 13% escalated. That is the
only routing result with numbers I found in TypeSafe's docs, so it is the pattern used here. Note the two
conditions their setup has: a 12x price gap, and a cheap model that is right 93% of the time.

Here: the small model answers, Jev reads `{"prompt", "response"}` and returns the probability that the answer
is wrong, and the large model answers when that reaches a threshold. A response with no `ANSWER:` line is
escalated without calling Jev (6% of test prompts).

### Check choice (dev only)

Three candidate checks, one request per dev response (`cache/jev_checks_dev.jsonl`):

| check | dev AUROC for "small answer is wrong" | 95% CI |
|---|---|---|
| `wrong_answer` | 0.937 | 0.901 to 0.971 |
| `flawed_reasoning` | 0.911 | 0.867 to 0.953 |
| `guess` | 0.813 | 0.752 to 0.869 |
| max of all three | 0.905 | 0.860 to 0.946 |
| response length (baseline) | 0.678 | 0.594 to 0.758 |

`wrong_answer` alone was kept. One check costs about 560 Jev input tokens ($0.024 per 1,000 queries).

### Result (test split, 200 prompts, 95% bootstrap CI)

- **The check works.** AUROC for spotting a wrong small answer: 0.927 (0.878 to 0.964), against 0.713 for
  response length; difference 0.215 (0.127 to 0.311). Small answers that pass at threshold 0.2 cover 54% of
  prompts and are 91.6% correct, against 58% for the small model overall.
- **Measured time (the real cost of local models).** Always-small 0.98 s, always-large 3.16 s.
  - Match-large threshold (0.28, cheapest dev point within 1 point of always-large): 76.5% in 2.81 s. That is
    11.2% less time (2.3 to 19.6) for 3.5 points less accuracy (-7.0 to -0.5). 43% escalated.
  - Best-gain threshold (0.86, the dev point furthest above the random line): 73.0% in 1.86 s. That is 41.2%
    less time (32.0 to 50.3) for 7.0 points less accuracy (-12.0 to -2.0), and 6.2 points above random
    routing at the same time (2.5 to 9.7). 20% escalated.
  - Averaged over every budget between always-small and always-large, the Jev cascade is 2.7 points above
    random routing (0.2 to 5.1). It is the only policy above zero: prompt length -2.9, Jev pre-routing -5.8,
    response-length cascade -4.8.
- **Imputed dollars.** No gain. Match-large threshold: 5.5% dearer than always-large (-12.6 to +1.3 saved).
  Averaged over budgets the cascade is 3.0 points below random (-5.5 to -0.5). One Jev check costs 31% of
  a large-model call at these prices, and the small model is paid for on every query.
- The cascade never reaches always-large accuracy cheaply. Where the small model is wrong, the large model is
  right only about 60% of the time, and those are its slowest prompts.

### Why this is weaker than the cookbook

Both of its conditions fail here. The cost gap is 3.2x in time and 5x in imputed dollars, not 12x. The small
model is right 58% of the time, not 93%, so nearly half of all prompts must be escalated to keep accuracy.

### Things tried and dropped

- `qwen3:8b` with thinking on, as a stronger "large": on 16 dev prompts it took 8.5x longer (32 s against
  3.8 s) and got the same 14 right. No accuracy gain, so it was not run in full.

### Honesty notes

- The cascade, the time view, the best-gain threshold and the "mean gain over random" measure were all added
  after the pre-routing test result was seen. The last one was added after looking at the test chart. Every
  threshold and the choice of check were made on dev only, and mean gain involves no threshold at all, but
  the decision about what to report was not made in advance. A fresh sample of prompts would settle it.
- Time is measured wall-clock on one laptop GPU, one run. It includes network time for Jev.
- `cascade_threshold` in `config.yaml` is 0.28. Use 0.86 to favour speed.

## Original proposal: Haiku 5.5 / Opus 5.5 on OpenRouter (not used)

Same family, both on OpenRouter, prices read from the OpenRouter model pages on 2026-11-12.

| Role | OpenRouter ID | USD per M input | USD per M output |
|---|---|---|---|
| small | `anthropic/claude-haiku-5.5` | 0.10 | 0.50 |
| large | `anthropic/claude-opus-5.5` | 4.00 | 20.00 |

Price ratio is 40x on both input and output.

Caveats:

- Prices vary by upstream provider on OpenRouter. Haiku is listed at $0.10/$0.50 with some providers at $0.11/$0.55.
  Opus is $4/$20 at most providers, $4.40/$22 at some, and one Anthropic listing shows $8/$40 (the page does not say what that tier is).
  Plan: record the actual `usage.cost` OpenRouter returns for every call and use that in the cost curve, with the
  table prices only for the pre-run estimate. Optionally pin the provider in the request so the rate is fixed.
- Both models accept `reasoning` parameters. The pages do not say whether reasoning is on by default. M1 will set it
  off explicitly for both so the comparison is plain small versus large and output tokens stay under `max_output_tokens`.
- `temperature: 0` support on these models is not confirmed from the pages. The smoke test will show it.

Worst-case cost estimate for the full run (400 prompts, about 300 input tokens, 600 output tokens each):

- Opus: 400 x (300 x $4 + 600 x $20) / 1e6 = about $5.30
- Haiku: 400 x (300 x $0.10 + 600 x $0.50) / 1e6 = about $0.13
- Jev: 400 x about 350 tokens x $0.042 / 1e6 = under $0.01
- Total about $5.45 worst case, under `max_usd: 10`. Realistic spend is lower since most answers stop well short of 600 tokens.

Risk to flag now: MMLU, ARC-Challenge and GSM8K are close to saturated for current models, so Haiku 5.5 may land
within 8 points of Opus 5.5 and trip the M4 gap check. If the pilot shows that, the options are a larger share of
MMLU-Pro plus competition math, or a weaker small model. Decision deferred to M4 as the spec says.
