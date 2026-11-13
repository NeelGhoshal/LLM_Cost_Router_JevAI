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
