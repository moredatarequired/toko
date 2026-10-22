#!/bin/sh
# Gate a CI job on probe output. Fails if a client this job uses accepted a
# compromised chain, or gave no ok verdict at all (no egress, interception).
# Other clients' VULNERABLE lines become warnings: they are runner-image tools
# this job never runs.
#
#   gate.sh PROBE_OUTPUT CLIENT...
set -u
out=$1
shift
status=0
alt=""
for c in "$@"; do
  alt="$alt${alt:+|}$c"
  if grep -E "^VULNERABLE +$c[ (]" "$out"; then status=1; fi
  if ! grep -qE "^ok +$c[ (]" "$out"; then
    echo "::error::$c gave no ok verdict, so its trust store is unverified"
    status=1
  fi
done
grep -E '^VULNERABLE' "$out" | grep -vE "^VULNERABLE +($alt)[ (]" | while read -r line; do
  echo "::warning::$line"
done
exit $status
