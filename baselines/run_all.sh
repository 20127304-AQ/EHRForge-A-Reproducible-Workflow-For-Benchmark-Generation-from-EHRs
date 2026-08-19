#!/usr/bin/env bash
set -euo pipefail

modal run scripts/01_retrieval.py --phase all
modal run scripts/02_readers.py --readers all --contexts all
modal run scripts/03_metrics.py --mode all --combinations all
modal run scripts/04_validate.py --strict
