#!/usr/bin/env bash
# Thin wrapper around the Makefile targets (install, test, baseline, adaptive, sweep, figures).
set -euo pipefail
cd "$(dirname "$0")"
make "$@"
