#!/usr/bin/env bash
# Fail if a machine-specific path, private host, secret or the private
# fine-tune's name leaks into the tree.
#
# The public tree is whatever git tracks. This walks it with `git grep`, prints
# every hit as file:line and exits 1. It is meant for CI and for the release
# checklist; run it from anywhere inside the repository.
#
#   bash scripts/check-public-tree.sh
#
# It also fails on the private fine-tune's name, which must not be published.
#
# Deliberate exclusions:
#   docs/reference/        upstream reference material quoted verbatim;
#   temp/                  scratch material deleted before publication;
#   this script            it has to contain the patterns it looks for.
#
# Documented install examples such as C:\rocm-sdk are allowed: none of the
# patterns below match them.
set -u

cd "$(git rev-parse --show-toplevel)" || exit 2

EXCLUDES=(
    ':!docs/reference'
    ':!temp'
    ':!scripts/check-public-tree.sh'
)

fail=0

check() {
    local label=$1 pattern=$2 hits
    hits=$(git grep -n -I -E -- "$pattern" -- "${EXCLUDES[@]}" 2>/dev/null)
    if [ -n "$hits" ]; then
        printf '%s\n' "$hits" | sed "s/^/[$label] /"
        fail=1
    fi
}

# Author-machine absolute paths.
check 'windows-user'  'C:\\Users\\'
check 'windows-model' 'C:\\models\\'
check 'windows-repo'  'C:\\hipml-'
check 'llama-cpp'     'C:\\llama\.cpp'
check 'posix-home'    '/Users/[A-Za-z0-9._-]+/'

# Private network addresses.
check 'private-net'   '192\.168\.'
check 'private-net'   '10\.0\.'

# Internal services and credentials.
check 'internal-host' 'wiki\.probackend'
check 'internal-host' 'agentbus'
check 'internal-host' 'openbao'
check 'secret'        'BAO_TOKEN'
check 'secret'        'ghp_[A-Za-z0-9]'
check 'secret'        '(^|[^[:alnum:]_])sk-[A-Za-z0-9]'
check 'secret'        'BEGIN [A-Z ]*PRIVATE KEY'

# The author's machine name, as a whole word so GPU brand names do not match.
check 'host-name'     '(^|[^[:alnum:]_])rtx([^[:alnum:]_]|$)'

# The private fine-tune the engine was developed against. It is not the model
# the release runs and must not be named; the release model is Qwen3.8-27B.
check 'private-model' '[Aa][Rr][Ee][Tt][Ee]'

if [ "$fail" -ne 0 ]; then
    echo
    echo "FAIL: machine-specific data found in the public tree (see above)." >&2
    exit 1
fi

echo "OK: no machine-specific paths, private hosts or secrets in the tracked tree."
exit 0
