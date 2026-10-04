#!/bin/sh
# Download a Pythia model from Hugging Face into models/<name>.
#
#   tools/fetch_pythia.sh               pythia-160m
#   tools/fetch_pythia.sh pythia-70m
set -eu

name=${1:-pythia-160m}
root=$(cd "$(dirname "$0")/.." && pwd)
dest="$root/models/$name"
mkdir -p "$dest"

for file in config.json tokenizer.json model.safetensors; do
  if [ -s "$dest/$file" ]; then
    echo "have $dest/$file"
  else
    echo "fetching $file"
    curl -fL --progress-bar -o "$dest/$file.part" "https://huggingface.co/EleutherAI/$name/resolve/main/$file"
    mv "$dest/$file.part" "$dest/$file"
  fi
done
