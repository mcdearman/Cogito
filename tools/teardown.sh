#!/bin/sh
# Tear a rented node down without anyone watching: wait for its work, copy the
# results here, destroy the instance. Start it detached, in the same step that
# starts the work, so that it outlives the session or agent that started it:
#
#   nohup tools/teardown.sh <ip> <machine id> <label> [max hours] \
#     > results/teardown-<machine id>.log 2>&1 &
#
# The node's work is done when /home/pool.done exists. After `max hours`
# (default 12) the node is destroyed whether or not the work is done, so that
# a hung run cannot bill for ever; whatever results exist are copied first.
# The node is never destroyed before a copy has been tried, and if the copy
# fails before the deadline it is tried again.
#
# Needs the `jl` CLI and an API key at ~/.config/jarvislabs/api_key.
set -u
ip=$1; id=$2; label=$3; hours=${4:-12}
root=$(cd "$(dirname "$0")/.." && pwd)
jl=${JL:-$HOME/.local/share/jarvislabs-venv/bin/jl}
deadline=$(( $(date +%s) + hours * 3600 ))
say() { echo "$(date -u +%Y-%m-%dT%H:%MZ) $*"; }
remote() { ssh -o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=20 "root@$ip" "$@"; }
destroy() {
  JL_API_KEY=$(cat "$HOME/.config/jarvislabs/api_key") JL_NO_UPDATE_CHECK=1 "$jl" destroy "$id" --yes --json 2>&1 | grep -q '"success": true'
}

say "watching $ip ($id), deadline in $hours hours"
finished=no
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ "$(remote '[ -f /home/pool.done ] && echo done' 2>/dev/null)" = "done" ]; then finished=yes; break; fi
  sleep 60
done
say "work finished: $finished"

copied=no
while :; do
  if "$root/tools/node.sh" "root@$ip" pull "$label" >/dev/null 2>&1; then copied=yes; break; fi
  [ "$(date +%s)" -ge "$deadline" ] && break
  say "copy failed; trying again"
  sleep 60
done
say "results copied: $copied"

n=0
until destroy; do
  n=$((n + 1))
  # Already gone counts as done.
  if ! JL_API_KEY=$(cat "$HOME/.config/jarvislabs/api_key") JL_NO_UPDATE_CHECK=1 "$jl" list --json 2>/dev/null | grep -q "\"machine_id\": $id"; then break; fi
  [ "$n" -ge 30 ] && { say "DESTROY FAILED for $id: destroy it by hand"; exit 1; }
  sleep 60
done
say "destroyed $id"
