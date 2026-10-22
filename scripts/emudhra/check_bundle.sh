#!/bin/sh
# Part of the emudhra-ca-remediation skill.
# Offline check: does a CA bundle still trust the compromised eMudhra
# intermediates?
#
#   check_bundle.sh [-c CHAINS_DIR] BUNDLE [BUNDLE ...]
#
# BUNDLE is a PEM file (certifi's cacert.pem, /etc/ssl/certs/ca-certificates.crt,
# a vendored bundle, a bundle copied out of an image), or the word "system"
# for OpenSSL's default store on this machine.
#
# It verifies the twelve chains served by the advisory's test hosts against
# each bundle. Any chain that verifies means the bundle still trusts a
# compromised intermediate. The chains come from
# https://emudhra.we-have-a-problem.com/chains.zip, downloaded and checked
# against a pinned SHA-256 unless -c points at an already-unpacked copy.
#
# A clean result covers the bundle only. Runtimes that block the
# intermediates in code (patched Go, Node, rustls-webpki, CPython) can be
# safe with an old bundle, and a runtime that ignores this bundle is not
# made safe by it. probe_runtime.sh tests the running client end to end.
#
# Exit status: 0 all bundles clean, 1 at least one trusts a chain, 2 usage
# or setup error, including a bundle that is missing, unreadable or not PEM.
# A setup error is never reported as CLEAN.

set -u
CHAINS_URL=https://emudhra.we-have-a-problem.com/chains.zip
CHAINS_SHA256=8a2edc81b6c51e3ad27d45ca1e441ec84c961e2dee0fc547f2a02c0f74315f95

chains=""
if [ "${1:-}" = "-c" ]; then
  chains=${2:?-c needs a directory}
  shift 2
fi
[ $# -ge 1 ] || { sed -n '3,25p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
command -v openssl >/dev/null 2>&1 || { echo "openssl is required" >&2; exit 2; }

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

if [ -z "$chains" ]; then
  tmp=$(mktemp -d) || exit 2
  trap 'rm -rf "$tmp"' EXIT
  if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$tmp/chains.zip" "$CHAINS_URL"
  else wget -q -O "$tmp/chains.zip" "$CHAINS_URL"; fi || { echo "could not download $CHAINS_URL" >&2; exit 2; }
  got=$(sha256 "$tmp/chains.zip")
  [ "$got" = "$CHAINS_SHA256" ] || { echo "chains.zip SHA-256 mismatch ($got); not using it" >&2; exit 2; }
  if command -v unzip >/dev/null 2>&1; then unzip -q "$tmp/chains.zip" -d "$tmp"
  else python3 -I -m zipfile -e "$tmp/chains.zip" "$tmp"; fi || { echo "could not unpack chains.zip" >&2; exit 2; }
  chains="$tmp/chains"
fi
ls "$chains"/*emudhra*.pem >/dev/null 2>&1 || { echo "no *emudhra*.pem chains in $chains" >&2; exit 2; }

# -no_check_time: the test chains expire, and an expired chain must still count as
# trusted. Use only the bundle given: no default directory (-no-CApath) and, on OpenSSL 3,
# no default store URI (-no-CAstore). LibreSSL has neither option.
only=""
vhelp=$(openssl verify -help 2>&1)
case "$vhelp" in *-no-CApath*) only="-no-CApath" ;; esac
case "$vhelp" in *-no-CAstore*) only="$only -no-CAstore" ;; esac

status=0
for bundle in "$@"; do
  if [ "$bundle" != system ]; then
    if [ ! -f "$bundle" ] || [ ! -r "$bundle" ]; then
      echo "ERROR      $bundle is not a readable file" >&2; status=2; continue
    fi
    if ! grep -q 'BEGIN CERTIFICATE' "$bundle"; then
      echo "ERROR      $bundle holds no PEM certificates (DER, a directory listing, or the wrong file?)" >&2
      status=2; continue
    fi
  fi
  trusted=0; total=0; hits=""; broken=""
  for chain in "$chains"/*emudhra*.pem; do
    total=$((total + 1))
    if [ "$bundle" = system ]; then
      out=$(openssl verify -no_check_time -untrusted "$chain" "$chain" 2>&1)
    else
      # shellcheck disable=SC2086
      out=$(openssl verify -no_check_time $only -CAfile "$bundle" -untrusted "$chain" "$chain" 2>&1)
    fi
    case "$out" in
      *": OK"*) trusted=$((trusted + 1)); hits="$hits $(basename "$chain" .we-have-a-problem.com.pem)" ;;
      *"error "[0-9]*" at "*) ;;  # a verification failure: the bundle rejected the chain
      *) broken=$(printf '%s\n' "$out" | head -1) ;;
    esac
  done
  if [ -n "$broken" ] && [ "$trusted" -eq 0 ]; then
    echo "ERROR      $bundle could not be checked: $broken" >&2
    status=2; continue
  fi
  if [ "$trusted" -gt 0 ]; then
    echo "VULNERABLE $bundle trusts $trusted/$total compromised chains:$hits"
    [ "$status" = 2 ] || status=1
  else
    echo "CLEAN      $bundle trusts 0/$total compromised chains"
  fi
done
exit $status
