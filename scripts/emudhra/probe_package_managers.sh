#!/bin/sh
# Part of the emudhra-ca-remediation skill.
# Canary check for package managers: point each one present (pip, uv, npm,
# pnpm, go, cargo, gem, mvn, composer) at the advisory's test hosts as its
# index or registry, and read which error comes back.
#
#   ok            a TLS/certificate error: the tool refused the chain
#   VULNERABLE    a 404 / "not found": the tool finished the handshake
#   INCONCLUSIVE  anything else, or TLS is intercepted (see probe_runtime.sh)
#
# This tests each tool's own TLS stack, which is not always the OS store:
# pip adds its vendored certifi, uv ships webpki-roots, npm uses Node's
# bundled roots, mvn uses the JDK cacerts.
#
# Every tool runs with an empty temporary config, home and netrc, so no
# registry token is sent to the test host even by a vulnerable client.
# Run it on a laptop, a CI runner, or inside a build image:
#
#   docker run --rm -i --entrypoint sh BUILD_IMAGE -s < probe_package_managers.sh
#
# QUICK=1 probes one chain per root instead of all twelve.

set -u
have() { command -v "$1" >/dev/null 2>&1; }
# ONLY="uv pip" limits the run to those tools. Probing a tool runs it, and mvn fetches
# and executes a plugin over the TLS under test, so keep credential-bearing jobs to
# the tools they already use.
wanted() {
  [ -z "${ONLY:-}" ] && return 0
  case " $ONLY " in *" $1 "*) return 0 ;; esac
  return 1
}
want() { have "$1" && wanted "$1"; }
if [ "${QUICK:-0}" = 1 ]; then
  HOSTS="dv-g3b-emudhra.we-have-a-problem.com dv-g3b-aaa-emudhra.we-have-a-problem.com"
else
  HOSTS=""
  for kind in dv ov ev; do for gen in g3b g3c; do for via in "" -aaa; do
    HOSTS="$HOSTS $kind-$gen$via-emudhra.we-have-a-problem.com"
  done; done; done
fi

WORK=$(mktemp -d) || exit 2
trap 'rm -rf "$WORK"' EXIT
# A throwaway home: no ~/.npmrc, ~/.pypirc, ~/.netrc, ~/.m2/settings.xml,
# ~/.cargo/credentials, ~/.gem/credentials or ~/.composer/auth.json.
CLEAN="env -u UV_INDEX -u UV_DEFAULT_INDEX -u UV_FIND_LINKS HOME=$WORK/home XDG_CONFIG_HOME=$WORK/home/.config NETRC=$WORK/home/.netrc
  PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL= PIP_EXTRA_INDEX_URL= PIP_NO_INPUT=1
  UV_INDEX_URL= UV_EXTRA_INDEX_URL= UV_NO_CONFIG=1 UV_CACHE_DIR=$WORK/uv
  NPM_CONFIG_USERCONFIG=$WORK/home/.npmrc NPM_CONFIG_GLOBALCONFIG=$WORK/home/.npmrc-g NPM_CONFIG_CACHE=$WORK/npm
  CARGO_HOME=$WORK/cargo GOPATH=$WORK/go GOMODCACHE=$WORK/go/mod GOFLAGS=-mod=mod GOSUMDB=off GONOPROXY= GOPRIVATE=
  COMPOSER_HOME=$WORK/composer COMPOSER_AUTH= GEM_HOME=$WORK/gem"
mkdir -p "$WORK/home/.config" && : > "$WORK/home/.netrc"
printf '<settings/>\n' > "$WORK/m2-settings.xml"

# Interception check: who issued pypi.org's certificate? openssl s_client ignores
# HTTPS_PROXY, so fall back to curl, which goes through the proxy like the tools do.
# An issuer we can't read is treated as interception: a proxy that broke the
# check could also be the reason every tool "refused" the test hosts. Set
# INTERCEPTED=0 to override once you know egress is direct.
INTERCEPTED=${INTERCEPTED:-}
if [ -z "$INTERCEPTED" ]; then
  issuer=""
  [ -n "${HTTPS_PROXY:-}${https_proxy:-}" ] || ! have openssl || issuer=$(echo | openssl s_client -connect pypi.org:443 -servername pypi.org 2>/dev/null | openssl x509 -noout -issuer 2>/dev/null)
  [ -n "$issuer" ] || ! have curl || issuer=$(curl -sS -v -o /dev/null --max-time 20 https://pypi.org/ 2>&1 | sed -n 's/.*issuer: *//p' | head -1)
  case "$issuer" in
    *GlobalSign*|*"Let's Encrypt"*|*DigiCert*|*Sectigo*|*"Google Trust"*|*Amazon*|*Certainly*|*Entrust*|*"SSL.com"*) INTERCEPTED=0 ;;
    "") INTERCEPTED=1; echo "WARNING: could not read pypi.org's certificate issuer, so interception can't be ruled out; every result below is INCONCLUSIVE. Set INTERCEPTED=0 if egress is direct." ;;
    *) INTERCEPTED=1; echo "WARNING: TLS to pypi.org is intercepted ($issuer); every result below is INCONCLUSIVE." ;;
  esac
fi

judge() { # $1 tool, $2 host, $3 output file
  if [ "$INTERCEPTED" = 1 ]; then echo "INCONCLUSIVE $1 $2 (intercepted)"; return; fi
  # Network errors first, so a "TLS handshake timeout" is not read as a refusal.
  if grep -qiE 'Failed to establish a new connection|Name or service not known|name resolution|timed out|handshake timeout|Connection refused|ProxyError|ENOTFOUND|ECONNREFUSED|ETIMEDOUT|ECONNRESET|network is unreachable|no such host' "$3"; then
    echo "INCONCLUSIVE $1 $2: network error, not a TLS verdict"
  elif grep -qiE 'certificate|x509|PKIX|SSLCertVerificationError|CERTIFICATE_VERIFY_FAILED|UNABLE_TO_GET_ISSUER|UnknownIssuer|self.signed|invalid peer cert' "$3"; then
    echo "ok           $1 refused $2"
  elif grep -qiE '404|not found|E404|No matching distribution|no versions? (available|found)|unknown revision|Could not find artifact|could not find|failed to resolve|no matching package|does not exist|Not Found' "$3"; then
    echo "VULNERABLE   $1 reached $2: $(grep -iE '404|not found|no matching|could not find|unknown revision|does not exist' "$3" | head -1 | cut -c1-120)"
  else
    echo "INCONCLUSIVE $1 $2: $(tr '\n' ' ' < "$3" | cut -c1-160)"
  fi
}

TO=""
have timeout && TO="timeout 60"

for h in $HOSTS; do
  u="https://$h"
  o="$WORK/out"
  for py in python3 python; do
    if wanted pip && have $py && $py -m pip --version >/dev/null 2>&1; then
      $CLEAN $TO $py -m pip download --isolated --no-deps --no-cache-dir --retries 0 --timeout 15 \
        -d "$WORK/dl" --index-url "$u/simple/" emudhra-canary > "$o" 2>&1
      judge "$py -m pip ($($py -m pip --version | cut -d' ' -f2))" "$h" "$o"
      break
    fi
  done
  if want uv; then
    $CLEAN $TO uv pip compile --no-config --index-url "$u/simple/" - > "$o" 2>&1 <<EOF
emudhra-canary
EOF
    judge "uv ($(uv --version | cut -d' ' -f2))" "$h" "$o"
  fi
  if want npm; then
    $CLEAN $TO npm view emudhra-canary --registry "$u/" --fetch-retries 0 > "$o" 2>&1
    judge "npm (node $(node --version 2>/dev/null))" "$h" "$o"
  fi
  if want pnpm; then
    $CLEAN $TO pnpm view emudhra-canary --registry "$u/" > "$o" 2>&1
    judge "pnpm" "$h" "$o"
  fi
  if want go; then
    mkdir -p "$WORK/gomod" && printf 'module canary\n\ngo 1.21\n' > "$WORK/gomod/go.mod"
    (cd "$WORK/gomod" && $CLEAN GOPROXY="$u" $TO go mod download example.com/emudhra-canary@v1.0.0) > "$o" 2>&1
    judge "go ($(go env GOVERSION 2>/dev/null))" "$h" "$o"
  fi
  if want cargo; then
    mkdir -p "$WORK/cargo-proj/src" && : > "$WORK/cargo-proj/src/lib.rs"
    printf '[package]\nname = "c"\nversion = "0.0.0"\nedition = "2021"\n[dependencies]\nemudhra-canary = { version = "1", registry = "canary" }\n' > "$WORK/cargo-proj/Cargo.toml"
    (cd "$WORK/cargo-proj" && $CLEAN CARGO_REGISTRIES_CANARY_INDEX="sparse+$u/" CARGO_NET_RETRY=0 $TO cargo generate-lockfile) > "$o" 2>&1
    judge "cargo" "$h" "$o"
  fi
  if want gem; then
    $CLEAN $TO gem fetch emudhra-canary --clear-sources --source "$u/" > "$o" 2>&1
    judge "gem" "$h" "$o"
  fi
  if want mvn; then
    $CLEAN $TO mvn -B -q -s "$WORK/m2-settings.xml" -Dmaven.repo.local="$WORK/m2" dependency:get \
      -Dartifact=org.example:emudhra-canary:1.0 -DremoteRepositories="canary::default::$u/" > "$o" 2>&1
    judge "mvn (java $(java -version 2>&1 | grep -v 'Picked up' | head -1 | cut -d'"' -f2))" "$h" "$o"
  fi
  if want composer; then
    mkdir -p "$WORK/composer-proj"
    printf '{"repositories":[{"type":"composer","url":"%s"},{"packagist.org":false}],"require":{"emudhra/canary":"*"}}\n' "$u" > "$WORK/composer-proj/composer.json"
    (cd "$WORK/composer-proj" && $CLEAN $TO composer update --no-interaction --no-plugins --dry-run) > "$o" 2>&1
    judge "composer" "$h" "$o"
  fi
done
echo "== done. A tool marked vulnerable above still trusts a compromised intermediate."
echo "   Not covered: gradle, sbt, nuget, bundler, conda. Check those by runtime version (see https://emudhra.we-have-a-problem.com/)."
