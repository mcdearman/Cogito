#!/bin/sh
# Run sweeps on a rented GPU node over SSH.
#
#   tools/node.sh <host> setup               install everything on a fresh node
#   tools/node.sh <host> push                send this repository's HEAD to the node
#   tools/node.sh <host> sweep <config>...   start sweeps, each in its own tmux session
#                                            (<config>:<seed> runs one seed alone)
#   tools/node.sh <host> status              which sweeps are running, and their last lines
#   tools/node.sh <host> wait                block until no sweep is running
#   tools/node.sh <host> pull <label>        copy results into results/<label>/
#
# <host> is what ssh takes, such as root@203.0.113.7. The node needs an NVIDIA
# GPU, git, curl, g++, tmux and a Python with PyTorch (for its libtorch); a
# JarvisLabs PyTorch container has all of them. Everything is kept under /home,
# which is the part of a JarvisLabs container that survives a pause.
#
# A config is named without its directory or extension: `sweep m2-replay`.
# Sweeps started together share the one GPU; each needs about 3 GB of it.
set -eu

host=$1
action=$2
shift 2
root=$(cd "$(dirname "$0")/.." && pwd)
# The Meadow the node builds: the commit checked out in a local clone, sent
# over SSH, so that the node runs what this machine runs even when that is not
# on GitHub yet. MEADOW_SRC names the clone.
meadow_src=${MEADOW_SRC:-$root/../meadow}

remote() { ssh -o BatchMode=yes -o LogLevel=ERROR -o StrictHostKeyChecking=accept-new "$host" "$@"; }

case "$action" in
  setup)
    meadow_rev=$(git -C "$meadow_src" rev-parse HEAD)
    remote "mkdir -p /home/git && [ -d /home/git/meadow.git ] || git init -q --bare /home/git/meadow.git"
    GIT_SSH_COMMAND="ssh -o BatchMode=yes -o LogLevel=ERROR" git -C "$meadow_src" push -q -f "$host:/home/git/meadow.git" "$meadow_rev:refs/heads/node"
    remote "MEADOW_REV=$meadow_rev bash -s" <<'REMOTE'
set -eu
export CARGO_HOME=/home/.cargo RUSTUP_HOME=/home/.rustup MEADOW_HOME=/home/.meadow
[ -x /home/.cargo/bin/cargo ] || curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
export PATH=/home/.cargo/bin:/home/.meadow/bin:$PATH
[ -d /home/meadow ] || git clone -q /home/git/meadow.git /home/meadow
(cd /home/meadow && git fetch -q && git checkout -q "$MEADOW_REV" && scripts/install.sh --no-modify-path) > /home/setup-meadow.log 2>&1
[ -d /home/MeadowTorch ] || git clone -q https://github.com/mcdearman/MeadowTorch /home/MeadowTorch
(cd /home/MeadowTorch && git pull -q)
export LIBTORCH=$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')
PREFIX=/home/.local /home/MeadowTorch/shim/build.sh test 2>&1 | tail -2
PREFIX=/home/.local /home/MeadowTorch/shim/build.sh install 2>&1 | tail -1
mkdir -p /home/git /home/logs
[ -d /home/git/cogito.git ] || git init -q --bare /home/git/cogito.git
meadow --version
REMOTE
    "$0" "$host" push
    remote 'cd /home/cogito && tools/fetch_pythia.sh | tail -1 && tools/fetch_eval_data.sh | tail -1'
    ;;
  push)
    branch=$(git -C "$root" branch --show-current)
    GIT_SSH_COMMAND="ssh -o BatchMode=yes -o LogLevel=ERROR" git -C "$root" push -q -f "$host:/home/git/cogito.git" "$branch"
    remote "[ -d /home/cogito ] || git clone -q /home/git/cogito.git -b $branch /home/cogito; cd /home/cogito && git fetch -q && git checkout -q $branch && git reset -q --hard origin/$branch && git log --oneline -1"
    ;;
  sweep)
    # <config> runs every seed of the config in one process; <config>:<seed>
    # runs that seed alone, so that seeds can run side by side.
    for spec in "$@"; do
      name=${spec%%:*}
      seed=""
      session=$name
      if [ "$spec" != "$name" ]; then seed=${spec#*:}; session=$name-seed$seed; fi
      remote "tmux new-session -d -s $session 'export PATH=/home/.meadow/bin:\$PATH MEADOW_HOME=/home/.meadow; cd /home/cogito && meadow run . -- sweep configs/$name.json $seed > /home/logs/$session.log 2>&1; echo EXIT \$? >> /home/logs/$session.log' && echo started $session"
    done
    ;;
  status)
    remote 'tmux ls 2>/dev/null | cut -d: -f1 | tr "\n" " "; echo; for log in /home/logs/*.log; do echo "$(basename $log .log): $(tail -1 $log | cut -c1-100)"; done; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
    ;;
  wait)
    # In pieces, so that a dropped connection does not end the wait.
    while [ "$(remote 'n=0; while tmux ls >/dev/null 2>&1 && [ $n -lt 30 ]; do sleep 10; n=$((n+1)); done; tmux ls 2>/dev/null | wc -l')" != "0" ]; do :; done
    ;;
  pull)
    mkdir -p "$root/results/$1"
    rsync -a -e "ssh -o BatchMode=yes -o LogLevel=ERROR" "$host:/home/cogito/results/" "$root/results/$1/"
    rsync -a -e "ssh -o BatchMode=yes -o LogLevel=ERROR" "$host:/home/logs/" "$root/results/$1/logs/"
    ls "$root/results/$1"
    ;;
  *)
    echo "unknown action: $action" >&2
    exit 1
    ;;
esac
