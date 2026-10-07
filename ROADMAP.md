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
| Compute | Develop on an M2 Pro (16 GB, MPS); run sweeps on rented CUDA GPUs (JarvisLabs), set up by `tools/node.sh`. Nodes cost at most $1 an hour each; split runs across several, short runs on the cheapest |
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
| 2. Baselines | done |
| 3. Distillation, study notes, editing | done |
| 4. Memory layers and sparse memory finetuning | in progress |
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

Speed, measured on an A100 with one seed of the mixed distillation sweep (10,
100 and 1000 facts) alone on the GPU:

| | before | after |
| --- | --- | --- |
| whole sweep | 819 s | 259 s |
| training, 5000 steps | 652 s | 209 s |

"After" batches distillation's teacher and student passes, scores 32
questions per pass in evaluation, computes logits only at the positions that
are used, and has MeadowTorch look its functions up once. Sweeps before
commit 3298a47 ran the slower code, and shared a GPU, so their recorded
seconds are not comparable with later ones.

Known limits to revisit:

- Tests must run one at a time: `meadow test --test-threads 1`. Run in
  parallel, the Pythia tests end in a segmentation fault; the cause has not
  been established (several tests using MPS from different OS threads at once
  is the likely one).
- `Std.Ffi` copies bulk data element by element, so moving large tensors
  between Meadow and C is slow.
- Generation (the `complete` command) has no key/value cache and no
  batching. Evaluation does not generate, and is batched.
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

`meadow run . -- sweep configs/m2-full-ft.json` runs one method: for each seed
and each size `n` it starts from the untouched model, injects the first `n`
facts, and logs acquisition, retention and cost.

- [x] `src/Methods.mw`: the method interface, `inject method model tok facts`,
      which changes the model in place and answers its cost (steps, tokens,
      seconds, parameters trained, peak MPS memory, estimated FLOPs).
- [x] Full fine-tuning: AdamW on the language-model loss of every training
      statement, over every parameter.
- [x] LoRA: rank-8 adapters on every block's attention weights, or on its MLP
      weights as well, merged into the weights afterwards. Sweep not yet run.
- [x] Top layers only: the last two blocks, the final norm and the output
      embedding, with nothing below them differentiated, so the backward pass
      is truly shorter. Sweep not yet run.
- [x] Full fine-tuning with replay: at each step the loss of the statements is
      added to the loss of 8 windows of wikitext-2 train text (the analogue of
      biological interleaving). Sweep not yet run.
- [x] Retrieval (RAG) as a non-weight reference: stored statements matched to
      the prompt by rare shared words, the best two written before it. Sweep
      not yet run.

#### The frontier at 100 facts

Every trained method trades acquisition against forgetting through its
learning rate, so one setting each is not a fair comparison. One seed, 100
facts, 10 epochs in batches of 8, four learning rates a method; base model:
exact 0, perplexity 46.3 (`results/m2-frontier-*`).

| method | learning rate | exact, held-out prompts | perplexity | seconds |
| --- | --- | --- | --- | --- |
| replay | 5e-6 | 0.235 | 48.5 | 200 |
| | 1e-5 | 0.435 | 72.5 | 217 |
| | 2e-5 | 0.560 | 149 | 194 |
| | 5e-5 | 0.770 | 453 | 194 |
| full fine-tuning | 2e-6 | 0.115 | 111 | 74 |
| | 5e-6 | 0.355 | 172 | 77 |
| | 1e-5 | 0.580 | 326 | 76 |
| | 2e-5 | 0.660 | 1732 | 80 |
| LoRA, all weights | 5e-5 | 0.100 | 838 | 50 |
| | 1e-4 | 0.175 | 655 | 46 |
| | 2e-4 | 0.345 | 750 | 45 |
| | 5e-4 | 0.350 | 15560 | 44 |
| LoRA, attention only | 2e-4 | 0.070 | 1383 | 36 |
| | 5e-4 | 0.040 | 1035 | 35 |
| | 1e-3 | 0.000 | 2.4 million | 36 |
| top 2 layers | 2e-5 | 0.140 | 1443 | 41 |
| | 5e-5 | 0.290 | 3016 | 40 |
| | 1e-4 | 0.375 | 20367 | 40 |
| | 2e-4 | 0.550 | 1.7 million | 40 |
| retrieval, 2 passages | - | 0.125 | 46.3 | 0 |

- Replay is the best baseline: at any level of acquisition it forgets least
  (0.56 exact at perplexity 149, where plain fine-tuning needs 326 for 0.58),
  at about 2.5 times the time. An earlier version that averaged the
  statements and the rehearsed text into one loss was no better than plain
  fine-tuning, because the text outweighed the statements six to one
  (`results/m2-frontier-replay-joint-loss`).
- Plain full fine-tuning is second.
- LoRA and top-layers learn less and forget more than full fine-tuning here.
  Attention-only LoRA fits its training statements but the held-out prompts
  barely benefit. This is the opposite of the usual "LoRA forgets less", and
  there is no reference implementation to check against, so treat it as
  provisional. Possible reasons, untested: Adam moves every adapter entry at
  the full rate on a small repeated dataset; the top-layers method trains the
  output embedding.
- Retrieval leaves the model untouched but this model reads its context badly.

Operating points for the saturation sweeps: full fine-tuning 1e-5, replay 2e-5
(which match at about 0.57 exact), LoRA on all weights 2e-4, top 2 layers 1e-4.

#### Saturation sweeps

10 epochs in batches of 8, mean over seeds 0 to 2 with [min, max], run on an
A100 with CUDA at commit 848746a (`results/a100/m2-*`). Exact is on held-out
prompts; base model: exact 0, perplexity 46.3.

| method | facts | exact | exact, two-hop | perplexity |
| --- | --- | --- | --- | --- |
| replay, 2e-5 | 10 | 0.667 [0.550, 0.850] | 0.111 | 48.8 |
| | 100 | 0.675 [0.625, 0.760] | 0.208 | 159 |
| | 1000 | 0.577 [0.552, 0.624] | 0.160 | 20522 |
| full fine-tuning, 1e-5 | 10 | 0.567 [0.400, 0.700] | 0.111 | 152 |
| | 100 | 0.618 [0.600, 0.655] | 0.225 | 350 |
| | 1000 | 0.647 [0.614, 0.691] | 0.144 | 19909 |
| LoRA, all weights, 2e-4 | 10 | 0.383 [0.300, 0.450] | 0.222 | 575 |
| | 100 | 0.437 [0.410, 0.465] | 0.146 | 817 |
| | 1000 | 0.309 [0.225, 0.352] | 0.060 | 4089 |
| top 2 layers, 1e-4 | 10 | 0.483 [0.400, 0.550] | 0.056 | 83280 |
| | 100 | 0.500 [0.475, 0.515] | 0.138 | 18664 |
| | 1000 | 0.345 [0.275, 0.390] | 0.071 | 5.8 million |
| retrieval, 2 passages | 10 | 0.067 [0.050, 0.100] | 0.000 | 46.3 |
| | 100 | 0.117 [0.095, 0.130] | 0.108 | 46.3 |
| | 1000 | 0.123 [0.115, 0.131] | 0.088 | 46.3 |

- No trained method keeps general text at 1000 facts. Replay, which holds
  perplexity at 48.8 for 10 facts and 159 for 100, collapses to about 20,000
  at 1000, the same as plain fine-tuning.
- Acquisition on held-out wording stays between about 0.55 and 0.7 for the
  two full fine-tuning methods at every size; it does not rise with volume.
  LoRA and top-layers learn less and fall off at 1000.
- Two-hop questions stay at or below about 0.2 for every method.
- Retrieval does not degrade with volume and never forgets, but tops out near
  0.12 because this model reads its context badly.

Limits of this sweep:

- The number of training steps grows with the number of facts (50, 500,
  5000), so "more facts" and "more updates" are not separated here; the grid
  below separates them.
- Each method ran at one learning rate, chosen at 100 facts.
- The five sweeps shared one GPU, so their wall-clock times are not
  comparable and are left out. Alone, a 100-fact full fine-tuning run trains
  in 19 s on the A100 and about 80 s on the M2 Pro.
- An earlier run of the same configs on the M2 Pro (MPS) gave the same
  picture with different numbers, for example full fine-tuning at 1000 facts
  0.516 [0.263, 0.694] against 0.647 here. Runs are not bit-reproducible
  across devices, or between two runs on MPS.

#### Facts against updates

The sweep above takes ten passes over the statements, so its steps grow with
its facts. This grid fixes the steps instead: 10, 100 and 1000 facts, each
trained for exactly 50, 500 and 5000 steps in batches of 8. Mean over seeds 0
to 2, A100, commit 4172b8b (`results/a100/m2-steps-*`). A fact has four
statements, so the passes over them are 10 x steps / facts.

Perplexity on general text (base 46.3):

| | 50 steps | 500 steps | 5000 steps |
| --- | --- | --- | --- |
| full fine-tuning, 10 facts | 152 | 229 | 6698 |
| full fine-tuning, 100 facts | 135 | 350 | 11816 |
| full fine-tuning, 1000 facts | 137 | 537 | 19909 |
| replay, 10 facts | 48.8 | 211 | 12572 |
| replay, 100 facts | 55.6 | 159 | 14188 |
| replay, 1000 facts | 54.8 | 152 | 20522 |

Exact on held-out prompts:

| | 50 steps | 500 steps | 5000 steps |
| --- | --- | --- | --- |
| full fine-tuning, 10 facts | 0.567 | 0.550 | 0.617 |
| full fine-tuning, 100 facts | 0.095 | 0.618 | 0.808 |
| full fine-tuning, 1000 facts | 0.017 | 0.030 | 0.647 |
| replay, 10 facts | 0.667 | 0.533 | 0.450 |
| replay, 100 facts | 0.093 | 0.675 | 0.722 |
| replay, 1000 facts | 0.018 | 0.032 | 0.577 |

- Forgetting follows the number of updates, not the number of facts. Read
  down a column and perplexity barely moves; read along a row and it climbs
  by orders of magnitude. Ten facts trained for 5000 steps wreck the model as
  surely as a thousand do.
- Acquisition follows the passes over each fact. With about ten passes
  (the diagonal) a method learns 0.55 to 0.68; with one pass (below the
  diagonal) almost nothing, 0.02 to 0.10. A hundred passes help 100 facts
  (0.81) but not 10.
- Together these are the saturation result for gradient baselines: learning a
  fact takes a roughly fixed number of updates, every update costs general
  knowledge whatever it teaches, so the damage grows with the number of facts
  and the model is gone by a few thousand steps.
- Replay delays the damage (48.8 against 152 at 50 steps) but does not stop
  it. At 5000 steps it has rehearsed its 1170 windows of general text about
  34 times each, so it may be overfitting them; a larger rehearsal set is
  untested.
- All of this is at one constant learning rate per method, with no warm-up,
  decay or weight decay.

#### 10,000 facts

Mean over seeds 0 to 2, A100 (`results/a100/m2-10k-*`).

| method | steps | exact | exact, two-hop | perplexity |
| --- | --- | --- | --- | --- |
| retrieval, 2 passages | 0 | 0.141 | 0.096 | 46.3 |
| full fine-tuning, one pass | 5000 | 0.015 | 0.001 | 25557 |
| replay, one pass | 5000 | 0.015 | 0.001 | 20407 |
| full fine-tuning, ten passes | 50000 | 0.541 [0.480, 0.578] | 0.109 | 3.1 billion |
| replay, ten passes | 50000 | 0.468 [0.377, 0.560] | 0.082 | 472,000 |

One pass over 10,000 facts teaches almost none of them and still destroys
general text, as the grid predicts for 5000 steps. Ten passes teach about
half of them, slightly less than at smaller sizes, and leave a model that
cannot model general text at all; replay's is less far gone but just as
unusable. For the gradient baselines 10,000 facts is past saturation: the
facts can be put in only by giving up everything else. Retrieval is unchanged by
volume: 0.067, 0.117, 0.123 and 0.141 at 10, 100, 1000 and 10,000 facts.

#### Larger models

The untouched 160M model reads a fact from its context badly, which is behind
the weak retrieval and the failure of pure distillation. Untouched Pythias on
the milestone 1 harness, three seeds (`results/m3b-partial/m1-base-*`):

| model | perplexity | closed book, exact | fact in prompt, exact | cloze prompts | two-hop |
| --- | --- | --- | --- | --- | --- |
| Pythia-160M | 46.3 | 0.000 | 0.154 | 0.267 | 0.208 |
| Pythia-410M | 29.1 | 0.000 | 0.316 | 0.533 | 0.267 |
| Pythia-1B | 23.9 | 0.004 | 0.381 | 0.616 | 0.396 |

Reading from context doubles from 160M to 410M and improves a little more at
1B. The same code runs all three.

Pythia-410M with facts injected, on an A30, mean over seeds 0 to 2
(`results/a30/b410-*`; base perplexity 29.1). Learning rates were chosen from
a one-seed frontier at 100 facts.

| method | facts | exact, held-out | exact, two-hop | perplexity |
| --- | --- | --- | --- | --- |
| full fine-tuning, 5e-6 | 100 | 0.847 [0.805, 0.870] | 0.250 | 71.0 |
| | 1000 | 0.885 [0.830, 0.930] | 0.219 | 205 |
| mixed distillation, 2e-5, no context | 100 | 0.687 [0.660, 0.705] | 0.208 | 63.1 |
| | 1000 | 0.741 [0.734, 0.748] | 0.194 | 114 |
| retrieval, 2 passages | 100 | 0.240 | 0.150 | 29.1 |
| | 1000 | 0.247 | 0.128 | 29.1 |

- The larger model learns facts far better. Full fine-tuning reaches 0.85 to
  0.89 on reworded prompts, where the 160M model stayed near 0.6. Much of the
  gap between storing a statement and knowing a fact was the model's size.
- It also forgets far less for it. At 1000 facts perplexity is 205, seven
  times its base, where the 160M model's was about 20,000, over four hundred
  times its base. The learning rate is half the 160M sweep's, so this is not
  a like-for-like comparison of sizes; at 5e-6 and 100 facts the 160M model
  had 0.355 exact at perplexity 172.
- Mixed distillation still forgets least, 114 at 1000 facts, but learns less
  at these settings (0.74 against 0.89), so the two no longer rank as they
  did on the small model.
- Two-hop questions do not improve with size: 0.19 to 0.25.
- Retrieval doubles to about 0.24, in line with the better context reading.
- Not yet run on 410M: more than 1000 facts, the step-budget grid, editing,
  and memory layers.

#### Full fine-tuning

Full fine-tuning is very sensitive to its learning rate. One seed, 100 facts,
10 epochs in batches of 8 (base model: exact 0, perplexity 46.3):

| learning rate | exact, held-out prompts | perplexity on general text |
| --- | --- | --- |
| 5e-6 | 0.335 | 157.6 |
| 1e-5 | 0.575 | 339.4 |
| 2e-5 | 0.710 | 1772 |
| 5e-5 | 0.725 | 65286 |

The same collapse happens on the CPU, so it is the method and not the MPS
backend. The baseline uses 1e-5. Because acquisition and forgetting trade off
through this one setting, methods have to be compared at equal acquisition, or
as whole curves, not at one setting each.

Full fine-tuning sweep, learning rate 1e-5, 10 epochs, batches of 8, mean over
seeds 0 to 2 with [min, max] (`results/m2-full-ft`, commit a3c7646):

| facts | exact, held-out prompts | exact, two-hop | perplexity | question nll | seconds |
| --- | --- | --- | --- | --- | --- |
| 0 (base) | 0.000 | 0.000 | 46.3 | 4.99 | - |
| 10 | 0.617 [0.500, 0.700] | 0.111 | 153 [148, 157] | 6.53 | 9 |
| 100 | 0.638 [0.620, 0.650] | 0.204 | 354 [343, 375] | 7.31 | 93 |
| 1000 | 0.516 [0.263, 0.694] | 0.134 | 22315 [18686, 28946] | 13.19 | 805 |

- Acquisition on held-out wording stays near 0.6 and does not improve with
  more facts; at 1000 it varies a lot by seed (0.26 to 0.69).
- Forgetting grows with volume: perplexity triples at 10 facts, is eight times
  the base at 100, and at 1000 the model no longer models general text at
  all. The number of steps grows with the number of facts, so this sweep
  cannot separate "more facts" from "more updates".
- Two-hop questions stay low (0.11 to 0.20): facts that are learned are mostly
  not composed.
- Peak MPS memory was 2.6 GB at every size.

One 10-fact configuration gave 0.600 in a trial and 0.700 in the sweep, so
results are not bit-reproducible on MPS; the cause has not been established.

FLOPs are analytic estimates (two per weight per token forward, twice that
backward), not measurements.

### 3. Consolidation methods

- [x] Context distillation (`distill` in `src/Methods.mw`).
- [x] Editing weights directly, after MEMIT (`src/Memit.mw`).
- [x] Self-generated study notes (`notes` in `src/Methods.mw`).

Acquisition is now also scored in the wording the facts were injected in
("seen"), beside the held-out prompts. All runs below are on an A100.

#### Context distillation

A frozen copy of the model, the teacher, reads one statement of a fact and
then another; the student reads only the second and is trained towards the
teacher's whole distribution over each next token. "Mixed" adds the ordinary
loss on the tokens themselves, weighted equally.

Frontier at 100 facts, one seed (`results/a100/m3-frontier-distill*`; base:
exact 0, perplexity 46.3):

| variant | learning rate | exact, held-out prompts | perplexity |
| --- | --- | --- | --- |
| pure | 5e-6 | 0.000 | 54.0 |
| | 1e-5 | 0.000 | 61.3 |
| | 2e-5 | 0.005 | 78.5 |
| | 5e-5 | 0.020 | 449 |
| mixed | 5e-6 | 0.120 | 59.9 |
| | 1e-5 | 0.240 | 68.7 |
| | 2e-5 | 0.425 | 96.0 |
| | 5e-5 | 0.605 | 361 |

Saturation sweep for the mixed variant at 2e-5, ten passes, mean over seeds 0
to 2 with [min, max] (`results/a100/m3-distill-hard`):

| facts | exact, held-out | exact, seen | exact, two-hop | perplexity |
| --- | --- | --- | --- | --- |
| 10 | 0.283 [0.200, 0.400] | 0.967 | 0.056 | 101 [96, 106] |
| 100 | 0.447 [0.425, 0.465] | 0.917 | 0.179 | 106 [96, 120] |
| 1000 | 0.599 [0.509, 0.644] | 0.901 | 0.181 | 442 [342, 595] |
| 10,000 | 0.419 [0.385, 0.483] | 0.871 | 0.099 | 14227 [12859, 15271] |

- Pure distillation learns nothing here: the model reads its context too
  badly for its in-context behaviour to carry the fact.
- The mixed variant is the first method to keep general text through 5000
  steps: perplexity 442 at 1000 facts, where full fine-tuning and replay
  reach about 20,000 for similar acquisition (0.647 and 0.577). It needs no
  general text, only a second copy of the model.
- It delays the collapse and does not prevent it. At 10,000 facts and 50,000
  steps (`results/a100/m3-10k-distill-hard`) perplexity is about 14,000:
  far from full fine-tuning's 3.1 billion or replay's 472,000, and still a
  model that no longer handles general text.
- Its acquisition on held-out wording rises with volume, which no baseline's
  did.
- The teacher's context matters little. With a teacher that reads only what
  the student reads (`results/h200/m3-distill-no-context`, three seeds), the
  method does about as well: 0.427 exact at perplexity 81.6 for 100 facts
  (0.447 at 106 with context) and 0.537 at 507 for 1000 (0.599 at 442). So
  what protects general text is the model being held to its own earlier
  answers, and "context distillation" is the wrong name for what works here.
  The context may add some acquisition at 1000 facts; three seeds do not
  settle it.
- Untested: other weightings of the two losses, and other learning rates at
  1000 facts.

Facts against updates, for the mixed variant (`results/h200/m3-steps-distill-*`,
three seeds; the 5000-step, 1000-fact row is the sweep above):

| steps | facts | exact, held-out | perplexity |
| --- | --- | --- | --- |
| 5000 | 10 | 0.283 | 233 |
| 5000 | 100 | 0.550 | 295 |
| 5000 | 1000 | 0.599 | 442 |
| 10,000 | 1000 | 0.627 | 1139 |
| 20,000 | 1000 | 0.669 | 4779 |
| 50,000 | 1000 | 0.566 | 12072 |

Forgetting follows the number of updates here too, and climbs steadily with
no cliff: perplexity roughly doubles or triples each time the steps double.
Plain fine-tuning reaches about 20,000 at 5000 steps; this method is at
12,000 after 50,000, so it buys about ten times as many updates. For the
20,000- and 50,000-step rows, two seeds' full records were overwritten when
results were gathered from three machines; their numbers come from the runs'
printed logs, which lack the per-category detail. Runs for chosen seeds now
write a file each.

#### Study notes

Before training, the model reads one statement of a fact and is started on a
restatement ("In other words,"); what it writes is a note, kept if it still
names the fact's object. Statements and kept notes are then trained on as in
full fine-tuning, at 1e-5 (`results/h200/m3-notes`, three seeds).

| facts | notes kept | exact, held-out | perplexity |
| --- | --- | --- | --- |
| 100 | about 95 of 400 | 0.632 | 503 |
| 1000 | about 920 of 4000 | 0.645 | 22310 |

No better than full fine-tuning without notes (0.618 at 350 and 0.647 at
19,909). About a quarter of what the model writes still names the object, and
those notes are mostly the statement again. A model this small is not a
useful author of its own study material; SEAL's result is with a far larger
model and a learned policy for what to write.

#### Editing weights directly

Tuning at 100 facts, one seed (`results/a100/m3-memit-*`):

| blocks edited | regularisation | size limit | exact, held-out | exact, seen | perplexity |
| --- | --- | --- | --- | --- | --- |
| 0-2 | 15 | 0.75 | 0.100 | 0.47 | 47.2 |
| 1-3 | 5 to 50 | 0.75 | 0.085 | 0.47 | 47.0 |
| 0-1 | 15 | 0.75 | 0.065 | 0.44 | 47.2 |
| 1-2 | 15 | 0.75 | 0.060 | 0.44 | 47.2 |
| 2-4 | 5 to 150 | 0.75 | 0.065 | 0.35 | 47.0 |
| 2-4 | 1500 | 0.75 | 0.040 | 0.27 | 46.8 |
| 2-4 | 15000 (the paper's) | 0.75 | 0.000 | 0.01 | 46.4 |
| 4-6 | 150 | 0.75 | 0.065 | 0.24 | 47.0 |
| 0-2 | 15 | 1.5 | 0.055 | 0.38 | 55.2 |

- Editing leaves general text untouched (47 against a base of 46.3), where
  every trained method at least doubles perplexity at 100 facts.
- It stores a fact mostly in the wording it was fitted on: about half come
  back exactly in seen wording and a tenth at most in held-out wording.
- Earlier blocks do better, regularisation stops mattering below about 150,
  and the paper's 15,000 lets no edit take on this model.
- Facts about one subject have to be edited as one request. A subject's key
  does not depend on what is asked about it, so separate requests for a
  person's birthplace and employer each undo the other; the first version did
  that and stored almost nothing even in seen wording
  (`results/a100/memit-one-request-per-fact`).
- There is no reference implementation to check this against, and it differs
  from the paper (see `src/Memit.mw`), so the low generalization may be the
  implementation's and not the method's.

Saturation sweep at blocks 0-2, regularisation 15, mean over seeds 0 to 2
(`results/a100/m3-memit`):

| facts | exact, held-out | exact, seen | exact, two-hop | perplexity |
| --- | --- | --- | --- | --- |
| 10 | 0.000 | 0.600 | 0.000 | 46.4 |
| 100 | 0.072 | 0.417 | 0.000 | 47.4 |
| 1000 | 0.030 | 0.126 | 0.000 | 71.9 |

Editing saturates by a different route from training. General text is barely
touched, but the edits crowd each other out: recall in seen wording falls from
0.60 to 0.13 between 10 and 1000 facts.

### 4. Memory layers and sparse memory finetuning

- [x] What exists: Meta's memory-layer code is public
      (github.com/facebookresearch/memory, non-commercial licence) with no
      pretrained checkpoints, and no code for sparse memory finetuning was
      found. A later paper (arXiv 2604.05248) retrofits memory modules into a
      pretrained model, which is the route taken here.
- [x] A memory layer for Pythia (`memoryRead` in `src/Pythia.mw`): product
      keys, a few slots read at each position, their values summed by how
      well their keys match.
- [x] `src/Memory.mw`: attach an empty memory layer beside the MLP of chosen
      blocks, and sparse memory finetuning of its values, with slots chosen
      by TF-IDF against general text.
- [x] A memory whose keys are trained (`Memory.warm`): on general text, with
      the dense weights frozen. Still far from the papers' memory layers,
      which are trained with the model from the start.
- [ ] Saturation sweeps.
- [ ] MoE expert-gated updates (only routed experts receive updates).

First frontier: memory layers at blocks 3 and 7, 16,384 slots each, 16 read
at a position, SGD on the values, 100 facts, one seed
(`results/h200/m4-frontier-memory-*`; base perplexity 46.3).

| slots updated at a step | learning rate | exact, held-out | exact, seen | perplexity |
| --- | --- | --- | --- | --- |
| all that were read | 10 | 0.000 | 0.86 | 56.1 |
| all that were read | 30 | 0.025 | 0.90 | 87.2 |
| all that were read | 100 | 0.035 | 0.96 | 324 |
| top 2000 | 10 | 0.015 | 0.70 | 47.4 |
| top 2000 | 30 | 0.015 | 0.92 | 51.8 |
| top 2000 | 100 | 0.015 | 0.92 | 83.7 |
| top 200 | 100 | 0.000 | 0.27 | 47.2 |

- It memorises almost without damage: 0.92 of facts recalled in seen wording
  with perplexity at 51.8, and choosing slots by TF-IDF does cut forgetting at
  equal recall (51.8 against 87.2 when every slot read is updated).
- It does not generalise: held-out wording stays at 0.00 to 0.03. Like weight
  editing, and more sharply, it stores the phrasing and not the fact.
- Two explanations were tested and neither held (`results/a30/m4-*`, one seed,
  100 facts). Training the memory's keys, query and values on general text
  first, with the dense weights frozen, left held-out recall at 0.00 to 0.06
  (best: top 2000 slots, 0.055 held-out, 0.89 seen, perplexity 52.7 from a
  warmed base of 37.3). A coarser memory of 1024 slots did no better, trained
  or not. Why a retrofitted memory stores phrasing and not facts is open.
- Warming a memory on wikitext-2's train split lowers perplexity on its test
  split from 46.3 to about 37, so a warmed memory has its own baseline.

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

- GPU sweeps run on rented nodes with `tools/node.sh` (setup, push, sweep,
  wait, pull). Destroy the node when the sweeps are done.
- Run tests with `meadow test --test-threads 1`. The installed `meadow`
  (aa26e3b or later) has everything this needs.
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
