#!/bin/sh
# Part of the emudhra-ca-remediation skill.
# Live check, run where the code actually runs: a production container, a
# CI runner, a VM, a laptop. It needs nothing but sh, so it can be piped in:
#
#   kubectl exec -i POD -c CONTAINER -- sh -s < probe_runtime.sh
#   docker run --rm -i --entrypoint sh IMAGE@sha256:DIGEST -s < probe_runtime.sh
#   ssh host sh -s < probe_runtime.sh
#
# For every TLS client it finds (curl, wget, Python's ssl default store,
# Python's certifi bundle, node, bun, deno, ruby, php) it connects to the
# advisory's test hosts, whose leaf certificates come from compromised
# intermediates, and prints one line per client and host:
#
#   VULNERABLE    the client accepted a certificate issued by an
#                 "EM DV/OV/EV TLS CA" intermediate
#   ok            the client refused it
#   INCONCLUSIVE  no egress, a timeout, or a TLS-intercepting proxy, which
#                 means the answer describes the proxy and not this runtime
#
# All twelve test hosts are probed by default, because clients differ by
# chain: on macOS the three G3B chains pass while G3C is blocked (Apple
# soft-fails OCSP for G3B). QUICK=1 probes one chain per root instead.
#
# Interception is detected from the issuer of a control site (pypi.org).
# Override the guess with INTERCEPTED=0 or INTERCEPTED=1.
#
# It also prints the versions that decide trust (OS ca-certificates, pip,
# certifi, uv, node, java, deno, bun) and any environment variable that
# points a client at a different bundle.

set -u
T=10
if [ "${QUICK:-0}" = 1 ]; then
  # One chain through emSign ECC Root CA - G3, one through the Sectigo
  # AAA Certificate Services cross-sign.
  HOSTS="dv-g3b-emudhra.we-have-a-problem.com dv-g3b-aaa-emudhra.we-have-a-problem.com"
else
  HOSTS=""
  for kind in dv ov ev; do for gen in g3b g3c; do for via in "" -aaa; do
    HOSTS="$HOSTS $kind-$gen$via-emudhra.we-have-a-problem.com"
  done; done; done
fi
have() { command -v "$1" >/dev/null 2>&1; }

echo "== where"
echo "host: $(hostname 2>/dev/null || cat /etc/hostname 2>/dev/null)"
[ -r /etc/os-release ] && . /etc/os-release && echo "os: ${PRETTY_NAME:-unknown}"
uname -sm 2>/dev/null

echo "== trust store and versions"
if have dpkg-query; then
  dpkg-query -W -f='ca-certificates ${Version}\n' ca-certificates 2>/dev/null
  dpkg-query -W -f='ca-certificates-java ${Version}\n' ca-certificates-java 2>/dev/null
elif have apk; then apk info -v ca-certificates 2>/dev/null
elif have rpm; then rpm -q ca-certificates 2>/dev/null
fi
for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
  [ -r "$f" ] || continue
  echo "bundle $f: $(grep -c 'BEGIN CERTIFICATE' "$f") certs (run check_bundle.sh on it for an offline verdict)"
done
if [ -n "${CONDA_PREFIX:-}" ] && [ -r "$CONDA_PREFIX/ssl/cacert.pem" ]; then
  echo "conda env $CONDA_PREFIX: its own bundle $CONDA_PREFIX/ssl/cacert.pem ($(grep -c 'BEGIN CERTIFICATE' "$CONDA_PREFIX/ssl/cacert.pem") certs); check it with check_bundle.sh"
fi
for v in CONDA_PREFIX OPENSSL_X509_TEA_DISABLE SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE GIT_SSL_CAINFO PIP_CERT NODE_EXTRA_CA_CERTS NODE_OPTIONS NODE_USE_SYSTEM_CA UV_NATIVE_TLS UV_SYSTEM_CERTS JAVA_TOOL_OPTIONS HTTPS_PROXY https_proxy; do
  eval "val=\${$v:-}"
  # Redact URL credentials (user:pass@) and *password= values: this output ends up in CI logs.
  [ -n "$val" ] && echo "env $v=$(printf '%s' "$val" | sed -e 's#//[^/@[:space:]]*@#//***@#g' \
    -e 's#\([Pp]assword[A-Za-z]*=\)[^[:space:]]*#\1***#g' | cut -c1-160)"
done
PY=""
for p in python3 python; do have $p && { PY=$p; break; }; done
if [ -n "$PY" ]; then
  $PY - <<'EOF' 2>/dev/null
import sys, ssl
print("python %s, %s, default cafile %s" % (sys.version.split()[0], ssl.OPENSSL_VERSION, ssl.get_default_verify_paths().cafile))
for mod in ("pip", "certifi", "requests", "httpx", "truststore"):
    try:
        m = __import__(mod)
        print("  %s %s%s" % (mod, getattr(m, "__version__", "?"), (" " + m.where()) if mod == "certifi" else ""))
    except Exception:
        pass
try:
    from pip._vendor import certifi as vc
    print("  pip's vendored certifi %s" % vc.__version__)
except Exception:
    pass
EOF
fi
have uv && echo "uv $(uv --version 2>/dev/null)"
have node && echo "node $(node --version 2>/dev/null)"
have java && echo "java: $(java -version 2>&1 | grep -v 'Picked up' | head -2 | tr '\n' ' ')"
have deno && echo "deno $(deno --version 2>/dev/null | head -1)"
have bun && echo "bun $(bun --version 2>/dev/null)"
have go && echo "go toolchain: $(go version 2>/dev/null)"

# Each probe prints: TRUSTED <issuer or ?> | REJECTED <why> | ERROR <why>
probe_curl() {
  out=$(curl -sS -v -o /dev/null --max-time $T "https://$1/" 2>&1); rc=$?
  case $rc in
    0) # curl prints the whole issuer DN; keep only its CN, as the other probes do.
       cn=$(printf '%s\n' "$out" | sed -n 's/.*issuer: *//p' | head -1 | sed -n 's/.*CN *= *//p' | sed 's/[,;/].*//')
       echo "TRUSTED ${cn:-?}" ;;
    60|35|51|58|77|83) echo "REJECTED curl exit $rc" ;;
    *) echo "ERROR curl exit $rc" ;;
  esac
}
probe_wget() {
  wget -q -O /dev/null -T $T "https://$1/" >/dev/null 2>&1; rc=$?
  case $rc in 0|8) echo "TRUSTED ?" ;; 5) echo "REJECTED wget exit 5" ;; *) echo "ERROR wget exit $rc" ;; esac
}
PYPROBE='
import socket, ssl, sys
mode, host = sys.argv[1], sys.argv[2]
try:
    if mode == "certifi":
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    else:
        ctx = ssl.create_default_context()
except ImportError:
    print("SKIP"); sys.exit()
try:
    with socket.create_connection((host, 443), timeout=10) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as s:
            issuer = dict(x[0] for x in s.getpeercert()["issuer"])
            print("TRUSTED", issuer.get("commonName", "?"))
except ssl.SSLCertVerificationError as e:
    print("REJECTED", str(e.verify_message)[:100])
except Exception as e:
    print("ERROR", repr(e)[:140])
'
JSPROBE='
const tls = require("tls"); const host = process.argv[process.argv.length - 1];
const s = tls.connect({host, port: 443, servername: host, timeout: 10000}, () => {
  const c = s.getPeerCertificate(); console.log("TRUSTED " + ((c && c.issuer && c.issuer.CN) || "?")); s.end(); });
s.on("timeout", () => { console.log("ERROR timeout"); s.destroy(); });
s.on("error", e => { const c = String(e.code || e.message);
  console.log((/CERT|ISSUER|SELF_SIGNED|UNABLE_TO|BLOCKED|VERIFY|UNTRUSTED/i.test(c + e.message) ? "REJECTED " : "ERROR ") + c); });
'
RBPROBE='
require "net/http"
host = ARGV[0]
begin
  Net::HTTP.start(host, 443, use_ssl: true, open_timeout: 10, read_timeout: 10) do |h|
    cn = h.peer_cert.issuer.to_a.find { |k, _| k == "CN" }
    puts "TRUSTED " + (cn ? cn[1] : "?")
  end
rescue OpenSSL::SSL::SSLError => e
  puts "REJECTED " + e.message[0, 100]
rescue => e
  puts "ERROR " + e.message[0, 140]
end
'
PHPPROBE='
$h = $argv[1];
$ctx = stream_context_create(["ssl" => ["verify_peer" => true, "capture_peer_cert" => true, "SNI_enabled" => true, "peer_name" => $h]]);
$s = @stream_socket_client("ssl://$h:443", $errno, $err, 10, STREAM_CLIENT_CONNECT, $ctx);
if ($s) { $p = stream_context_get_params($s); $c = openssl_x509_parse($p["options"]["ssl"]["peer_certificate"]);
  echo "TRUSTED " . ($c["issuer"]["CN"] ?? "?"); }
else { $e = error_get_last()["message"] ?? $err;
  echo (preg_match("/certificate|SSL operation failed/i", $e) ? "REJECTED " : "ERROR ") . substr($e, 0, 120); }
'
DENOPROBE='
const host = Deno.args[0];
try { const c = await Deno.connectTls({hostname: host, port: 443}); await c.handshake(); console.log("TRUSTED ?"); c.close(); }
catch (e) { const s = String(e); console.log((/certificate|UnknownIssuer|invalid peer/i.test(s) ? "REJECTED " : "ERROR ") + s.slice(0, 120)); }
'

run_clients() { # $1 = host; prints "<client> <probe result>"
  if have curl; then echo "curl $(probe_curl "$1")"; elif have wget; then echo "wget $(probe_wget "$1")"; fi
  if [ -n "$PY" ]; then
    for mode in default certifi; do
      r=$($PY -c "$PYPROBE" $mode "$1" 2>&1 | tail -1)
      [ "$r" = SKIP ] || echo "python-ssl-$mode $r"
    done
  fi
  have node && echo "node $(node -e "$JSPROBE" "$1" 2>&1 | tail -1)"
  have bun && echo "bun $(bun -e "$JSPROBE" "$1" 2>&1 | tail -1)"
  if have deno; then f=$(mktemp); printf "%s" "$DENOPROBE" > "$f.ts"; echo "deno $(deno run -A "$f.ts" "$1" 2>&1 | tail -1)"; rm -f "$f" "$f.ts"; fi
  have ruby && echo "ruby $(ruby -e "$RBPROBE" "$1" 2>&1 | tail -1)"
  have php && echo "php $(php -r "$PHPPROBE" "$1" 2>&1 | tail -1)"
}

echo "== control (pypi.org)"
control=$(run_clients pypi.org)
printf '%s\n' "$control" | sed 's/^/control: /'
cissuer=""
if have openssl; then
  cissuer=$(echo | openssl s_client -connect pypi.org:443 -servername pypi.org 2>/dev/null | openssl x509 -noout -issuer 2>/dev/null)
fi
[ -n "$cissuer" ] || cissuer=$(printf '%s\n' "$control" | sed -n 's/^[a-z-]* TRUSTED //p' | grep -v '^?$' | head -1)
echo "control issuer: ${cissuer:-unknown}"
if [ -z "${INTERCEPTED:-}" ]; then
  case "$cissuer" in
    *GlobalSign*|*"Let's Encrypt"*|*DigiCert*|*Sectigo*|*"Google Trust"*|*Amazon*|*Certainly*|*Entrust*|*"SSL.com"*|*R1[0-4]*|*E[5-9]*|*WR[1-4]*|*WE[1-4]*) INTERCEPTED=0 ;;
    "") INTERCEPTED=1; echo "could not read the control issuer, so interception can't be ruled out; set INTERCEPTED=0 if egress is direct" ;;
    *) INTERCEPTED=1 ;;
  esac
fi
[ "$INTERCEPTED" = 1 ] && echo "WARNING: TLS looks intercepted (issuer above is not a public CA). Only a client that accepted an EM intermediate counts; everything else is INCONCLUSIVE. Re-run where egress is direct, or use check_bundle.sh."

echo "== compromised chains"
for h in $HOSTS; do
  run_clients "$h" | while read -r client verdict rest; do
    case "$verdict" in
      TRUSTED)
        case "$rest" in
          "EM "*" TLS CA"*|*"CN=EM "*" TLS CA"*|*"CN = EM "*" TLS CA"*) echo "VULNERABLE   $client accepted $h (issuer: $rest)" ;;
          "?") if [ "$INTERCEPTED" = 1 ]; then echo "INCONCLUSIVE $client connected to $h but the issuer is unknown behind a proxy"
               else echo "VULNERABLE   $client accepted $h"; fi ;;
          *) echo "INCONCLUSIVE $client reached $h through an interceptor (issuer: $rest)" ;;
        esac ;;
      REJECTED)
        if [ "$INTERCEPTED" = 1 ]; then echo "INCONCLUSIVE $client refused $h, but behind an interceptor ($rest)"
        else echo "ok           $client refused $h ($rest)"; fi ;;
      *) echo "INCONCLUSIVE $client $h: $verdict $rest" ;;
    esac
  done
done
echo "== done. A client marked vulnerable above still trusts a compromised intermediate."
echo "   Not probed here: Go and Rust binaries, the JVM, .NET. Check those by version (see SKILL.md)."
