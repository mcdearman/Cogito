# Cogito roadmap: saturation testing for knowledge consolidation methods

An experimental testbed for measuring how well different methods move new
knowledge into a language model's weights ("consolidation"), how much they can
absorb before they saturate or start forgetting, and how much compute each
method spends doing it.

This file started as a handoff from a research conversation with Claude
(claude.ai, October 2026). Paper names and numbers in the background section
come from memory or quick web searches and must be verified against the papers
before they are relied on or cited.

## Decisions

| Question | Decision |
| --- | --- |
| Language | Meadow, with PyTorch through [MeadowTorch](https://github.com/mcdearman/MeadowTorch) (a C shim over libtorch, called with `Std.Ffi`) |
| Base model | Pythia-160M (`EleutherAI/pythia-160m`); training data and checkpoints are public |
| Compute | Develop on an M2 Pro (16 GB, MPS). Keep everything device-agnostic so full sweeps can run on a CUDA GPU later |
| Fact generation | Templates only, fully seeded. No LLM API |
| Python | None. Building, running and testing need only Meadow, libtorch and curl. `tools/make_reference.py` records how the Hugging Face reference fixture was made |

Still open:

- Fact domains, and how many facts the largest sweep injects.
- Priority order of methods if time is limited.
- Whether Meadow needs a green-thread pin for FFI (`Ffi.pinned`). Until then
  every program runs with `threads = 1`.

## Status

| Milestone | State |
| --- | --- |
| 0. Foundations | done |
| 1. Synthetic facts and eval harness | done |
| 2. Baselines | not started |
| 3. Distillation, study notes, editing | not started |
| 4. Memory layers and sparse memory finetuning | not started |
| 5. Pre-backprop update router | not started |
| 6. Spiking-network side track | not started |

## Milestones

### 0. Foundations

What has to exist before milestone 1 can start. None of it was in the original
handoff, which assumed Python and Hugging Face.

- [x] MeadowTorch: tensors, autograd, SGD/AdamW, CPU/CUDA/MPS, scoped tensor
      memory.
- [x] MeadowTorch 0.2.0: read weights from a `.safetensors` file; attention,
      `sin` and `cos`; builds against a standalone libtorch, with no Python.
- [x] `tools/fetch_pythia.sh`: download Pythia-160M's config, tokenizer and
      weights with curl into `models/` (not in git).
- [x] `src/Fields.mw`: raising accessors over `Std.Json`, for
      `tokenizer.json`, `config.json` and fixtures.
- [x] `src/Tokenizer.mw`: Pythia's byte-level BPE. Token ids match Hugging Face
      on all 16 reference texts, and decode back to the text.
- [x] `src/Pythia.mw`: the GPT-NeoX forward pass, built from individual ops so
      later methods can reach every block and parameter. Next-token logits are
      within 0.05 of Hugging Face on CPU and MPS for the 3 reference prompts.
- [x] Greedy generation: 12 tokens after each reference prompt match Hugging
      Face exactly.

The reference is `tests/fixtures/pythia-160m.reference.json`, made once from
Hugging Face `transformers` by `tools/make_reference.py` and committed.

Known limits to revisit:

- Tests must run one at a time: `meadow test --test-threads 1`. Run in
  parallel, the Pythia tests end in a segmentation fault; the cause has not
  been established (several tests using MPS from different OS threads at once
  is the likely one).
- `Std.Ffi` copies bulk data element by element, so moving large tensors
  between Meadow and C is slow.
- Generation has no key/value cache and no batching: one prompt at a time,
  with the whole sequence recomputed for each new token.
- The tokenizer does not normalize to NFC, and classifies non-ASCII characters
  with an approximate table.

### 1. Synthetic facts and eval harness

Done: `meadow run . -- eval configs/m1-base.json` scores the untouched model.

- [x] `src/Templates.mw`, `src/Facts.mw`: a seeded world of invented people,
      cities, organizations and countries. Six training statements and two
      held-out prompts (one cloze, one question) per relation; a fact gets a
      random four of the statements.
- [x] Two-hop questions (country of birth; city of the employer), each naming
      the two facts it rests on.
- [x] Facts that conflict with what the model knows: a real country given
      another country's capital, continent or language. There are 90 at most
      (30 countries), one per group of 18 invented facts.
- [x] Facts are ordered so that the first `n` are a valid injection set for
      any `n`, and a hop counts once both its facts are in.
- [x] `src/Eval.mw`: acquisition, scored with one forward pass per question.
      `exact` (greedy decoding would start with the answer), `tokens` (fraction
      of answer tokens that are the first choice) and `nll` (mean negative
      log-likelihood of an answer token). `tokens` stands in for the F1 the
      handoff asked for.
- [x] Retention: perplexity on wikitext-2 test, and the same three scores on
      NaturalQuestions-open after four worked examples.
- [x] A positive control: the same questions with the facts written before the
      prompt.
- [x] `src/Run.mw`: every result is a JSON line in
      `results/<name>/metrics.jsonl` with the time, git commit, config and seed.

Base model, Pythia-160M, 190 facts and 80 hops per seed, seeds 0 to 2, mean
[min, max]:

| | closed book | facts in the prompt |
| --- | --- | --- |
| exact, all facts | 0.000 [0.000, 0.000] | 0.154 [0.145, 0.161] |
| exact, cloze prompts | 0.000 | 0.267 [0.247, 0.279] |
| exact, question prompts | 0.000 | 0.040 [0.032, 0.047] |
| exact, two-hop | 0.000 [0.000, 0.000] | 0.208 [0.194, 0.231] |
| tokens right, all facts | 0.044 [0.029, 0.059] | 0.608 [0.598, 0.619] |
| nll, all facts | 8.376 [8.319, 8.417] | 2.689 [2.614, 2.744] |

Retention baseline: perplexity 46.3 on 16 windows of 256 tokens; general
questions 0.010 exact, nll 4.986, on 100 questions.

What these say, and their limits:

- Closed book, the model gets no fact exactly right in any category, which is
  the "about 0" this milestone asked for.
- The control moves every score, so the harness can see knowledge, but its
  ceiling is low: Pythia-160M copies an invented multi-token name out of its
  context exactly only about a quarter of the time on cloze prompts, and
  almost never on question prompts, where it starts a sentence instead of
  giving the name. Retrieval will therefore be a weak upper reference for this
  model, and `nll` and `tokens` are the scores to compare across methods.
- General-question accuracy is 1 in 100, too low to show forgetting. Use the
  question `nll` and the perplexity for retention.
- Not here yet: the injection sweep itself and the method interface, which
  arrive with the first baseline in milestone 2.

### 2. Baselines, with full saturation sweeps

- Full fine-tuning.
- LoRA.
- Top-k layers only (backward pass truncated, so a real compute saving).
- Full fine-tuning with replay of general data (the analogue of biological
  interleaving).
- Retrieval (RAG) as a non-weight upper reference.

### 3. Consolidation methods

- Context distillation: fact in context, train to answer without it.
- Self-generated study notes (SEAL-style rephrasings and implications).
- Locate-then-edit (ROME / MEMIT).

### 4. Memory layers and sparse memory finetuning

- A memory-layer model. Check whether Meta released code or checkpoints;
  otherwise implement a small product-key memory layer.
- Sparse memory finetuning; try to reproduce the reported learning/forgetting
  trade-off at small scale.
- MoE expert-gated updates (only routed experts receive updates).

### 5. Pre-backprop update router (the novel method)

A cheap scoring pass picks which blocks or slots should change, and the
backward pass runs only through those. Compare learning, forgetting and actual
compute against sparse memory finetuning.

### 6. Spiking-network side track

Simulations with three-factor, e-prop and burst-dependent rules on small tasks,
with efficiency proxies (synaptic operations, activity sparsity).

## Research questions

1. For each consolidation method, how does new-knowledge acquisition scale with
   the number of facts injected, and where does it saturate?
2. How much does each method degrade existing knowledge as injection volume
   grows?
3. What is the compute cost per fact learned (FLOPs, wall clock, fraction of
   parameters touched, peak memory)?
4. Can a router that chooses update locations *before* the backward pass match
   sparse memory finetuning's learning/forgetting trade-off while saving
   compute?
5. (Later) Do biologically inspired local rules, simulated as spiking networks,
   show different saturation behavior, and what do efficiency proxies predict
   for them?

## Experimental protocol

### Saturation sweep

- Inject facts in increasing batches (10, 100, 1k, 10k, 100k) or as a stream.
- After each step, measure acquisition, retention, and retention of earlier
  injected facts (forgetting within the stream).
- Plot acquisition and retention curves per method, against the ~2 bits per
  parameter capacity reference.

### Efficiency metrics

- FLOPs per update (forward and backward separately), wall clock, peak memory.
- Fraction of parameters changed per update and in total.
- Estimated memory traffic (bytes read and written) as a von Neumann bottleneck
  proxy.
- For spiking simulations: synaptic operations and activity sparsity
  (NeuroBench-style).

### Method interface

Every method implements the same interface, so sweeps are uniform:
`inject(model, facts, budget) -> (model, costReport)`.

## Working conventions

- Run tests with `meadow test --test-threads 1`.
- Fix seeds; report mean and spread over at least 3 seeds for headline results.
- Record design choices and why in `notes/decisions.md` once experiments begin.
- Verify paper claims from the background section before citing them.
- Raw logs and generated data stay out of git; they are reproducible from
  config and seed.

## Background

**The core gap.** Fine-tuning a model on an explanation teaches it to predict
the text, not necessarily to use the idea. Models apply knowledge in context
much better than knowledge in weights. Naive fine-tuning on new facts also
causes catastrophic forgetting. Brains learn continuously, cheaply and without
catastrophic forgetting, partly by consolidating short-term memory into
long-term memory and by making local, sparse updates. LLMs have a context
window (working memory) and weights (long-term memory) but no good process for
moving knowledge from one to the other.

**The distinction this project is aimed at.** Most "sparse update" methods
still run a full backward pass and mask the update. They reduce interference,
not compute. A method that decides where to route an update *before* backprop,
and skips the rest, could save both.

**Biological inspiration.**

- Complementary learning systems: the hippocampus learns fast and specifically,
  the neocortex slowly and generally. Sleep replay (sharp-wave ripples,
  time-compressed, coordinated with cortical slow oscillations and spindles)
  interleaves new memories with old ones during consolidation.
- Replay is not the only route: synaptic consolidation (LTP), non-hippocampal
  memory (skills, conditioning), and fast schema-based cortical learning (Tse
  et al. 2007) also exist. New information that fits existing structure is
  absorbed faster.
- Credit assignment without backprop: three-factor rules (Hebbian plus a
  neuromodulator), dendritic compartments carrying error or feedback signals
  (Wright, Hedrick & Komiyama, Science 2025), burst-dependent plasticity,
  behavioral timescale synaptic plasticity (one-shot), and "prospective
  configuration" (Oxford, Nature Neuroscience 2024).

**Relevant ML work.**

- Context distillation (Anthropic 2021; Snell et al. 2022), Constitutional AI's
  SL phase, Llama 2 safety context distillation, OpenAI deliberative alignment.
- Self-generated study material: SEAL (MIT 2025), synthetic continued
  pretraining / EntiGraph.
- Knowledge capacity: Allen-Zhu & Li, "Physics of Language Models": about 2
  bits of knowledge per parameter; facts need varied phrasings during training
  to be extractable.
- Knowledge injection: Ovadia et al. 2023 (fine-tuning vs retrieval); Gekhman
  et al. 2024 (fine-tuning on new knowledge increases hallucination).
- Editing: ROME, MEMIT (locate-then-edit), MEND (learned gradient transform),
  SERAC (scope classifier plus side memory), GRACE, WISE. Editing degrades as
  edits accumulate.
- Sparse updates: sparse memory finetuning (Lin et al., Meta, arXiv
  2510.15103). Updates only memory-layer slots highly activated by new data
  relative to pretraining usage (TF-IDF-style). Reported held-out
  NaturalQuestions F1 drop: 89% full fine-tuning, 71% LoRA, 11% sparse memory
  finetuning, at equal new-knowledge acquisition. The closest existing work to
  this project's central idea.
- Compute-saving sparsity: MoE (Switch Transformer, DeepSeek-V3), memory layers
  (Berges et al. 2024), mixture-of-depths. Works best when routing is learned
  during training. LoRA mostly saves memory, not compute ("LoRA Learns Less and
  Forgets Less").
- Between global backprop and local rules: block-local losses, delayed or
  decoupled backprop, forward-only fine-tuning (MeZO), low precision (FP8).
- Neuromorphic angle: local rules (three-factor, e-prop, burst-dependent,
  equilibrium propagation) can run on Loihi, SpiNNaker or analog chips. They
  can be simulated on GPUs (snnTorch, Norse, Brian2, Intel Lava) and their
  efficiency estimated with hardware-independent proxies, as standardized by
  NeuroBench. Proxies can miss real overheads; validate on hardware.
