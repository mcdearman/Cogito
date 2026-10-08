# Cogito

How much new knowledge can you put into a language model's weights before it
stops absorbing it, or forgets what it knew? Cogito is a testbed for that
question: it injects invented facts into a small model with one method after
another, under identical conditions, and measures what was learned, what was
lost, and what it cost.

It is written in [Meadow](https://github.com/meadow-lang/meadow), with PyTorch
through [MeadowTorch](https://github.com/meadow-lang/MeadowTorch). The model is
Pythia-160M, run from a tokenizer and forward pass written in Meadow and
checked against Hugging Face `transformers`.

## What it has found so far

For gradient-based baselines on Pythia-160M:

- **Forgetting follows the number of updates, not the number of facts.**
  Training on 10 facts for 5000 steps wrecks the model's grasp of general text
  as surely as training on 1000 facts does.
- **Learning a fact takes a roughly fixed number of passes over it.** About ten
  passes give 0.55 to 0.68 exact match on prompts worded differently from the
  training text; one pass gives almost nothing.
- **So the damage grows with the number of facts.** Rehearsing general text
  while training (replay) delays the collapse but does not prevent it, and at
  10,000 facts every trained baseline has lost general text entirely.
- **Retrieval never forgets**, and does not degrade with volume, but a model
  this small reads its context badly: about 0.14 exact match.

[ROADMAP.md](ROADMAP.md) has the numbers, the methods, their limits, and what
comes next. These are results for one small model at fixed learning rates, and
several are marked provisional there.

## How it works

- **Facts** are generated from a seed: invented people, cities, organizations
  and countries, each fact with several training statements and held-out
  prompts in other words. Some questions need two facts; some facts contradict
  what the model knows about real countries.
- **Methods** share one interface, `inject method model tok facts`, which
  changes the model and reports its cost. So far: full fine-tuning, LoRA,
  training only the top layers, fine-tuning with replay, and retrieval.
- **Evaluation** scores acquisition on the held-out prompts, and retention as
  perplexity on wikitext-2 and answers to NaturalQuestions.
- **Runs** are driven by JSON configs and append one JSON line per result,
  with the config, seed and git commit.

## Running it

You need a Meadow new enough to have `Std.Ffi` (a nightly, or a build from
source), a C++20 compiler, and curl. There is no Python.

```sh
# Build the MeadowTorch shim once; on aarch64 macOS this downloads libtorch.
git clone https://github.com/meadow-lang/MeadowTorch
MeadowTorch/shim/build.sh install

git clone https://github.com/mcdearman/Cogito && cd Cogito
tools/fetch_pythia.sh              # Pythia-160M, which the tests use (375 MB)
tools/fetch_pythia.sh pythia-410m  # the main model (910 MB)
tools/fetch_eval_data.sh       # wikitext-2 and NaturalQuestions pages

meadow test --test-threads 1                           # tests must run one at a time
meadow run . -- complete "The capital of France is"
meadow run . -- eval configs/m1-base.json              # score the untouched model
meadow run . -- sweep configs/m2-full-ft.json          # inject facts and measure
```

Results go to `results/<config name>/metrics.jsonl`. On a Mac the model runs on
MPS; `tools/node.sh` sets up a rented CUDA machine and runs sweeps there.

## Layout

| | |
| --- | --- |
| `src/Tokenizer.mw`, `src/Pythia.mw` | the tokenizer and the model |
| `src/Templates.mw`, `src/Facts.mw` | the fact generator |
| `src/Methods.mw` | the injection methods |
| `src/Eval.mw` | acquisition, retention and retrieval |
| `src/Run.mw` | evaluation runs, sweeps and logging |
| `configs/` | one JSON file per run |
| `tests/fixtures/` | reference outputs from Hugging Face |
| `tools/` | fetching the model and data; running on a GPU node |

## License

BSD 3-Clause. See [LICENSE](LICENSE).
