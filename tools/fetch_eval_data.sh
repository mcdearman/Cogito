#!/bin/sh
# Download the general-knowledge data retention is measured on, as JSON pages
# from the Hugging Face dataset server, into data/ (not in git).
#
#   wikitext-2 test        general text, for perplexity
#   nq_open validation     general questions
#   nq_open train          a few worked examples to show the model the format
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/data"

# page <file> <dataset> <config> <split> <offset>
page() {
  if [ -s "$root/data/$1" ]; then
    echo "have data/$1"
  else
    echo "fetching data/$1"
    curl -fsS -o "$root/data/$1.part" \
      "https://datasets-server.huggingface.co/rows?dataset=$2&config=$3&split=$4&offset=$5&length=100"
    mv "$root/data/$1.part" "$root/data/$1"
  fi
}

for offset in 0 100 200 300; do
  page "wikitext2-test-$offset.json" Salesforce/wikitext wikitext-2-raw-v1 test $offset
done
for offset in 0 100; do
  page "nq-open-validation-$offset.json" google-research-datasets/nq_open nq_open validation $offset
done
page "nq-open-train-0.json" google-research-datasets/nq_open nq_open train 0
