#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="qstn_dev"
PUSH_TAG=0

usage() {
    cat <<'EOF'
Usage:
  ./scripts/release.sh X.Y.Z [--push]

Examples:
  ./scripts/release.sh 0.2.2
  ./scripts/release.sh 0.2.2 --push
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 1
fi

VERSION="$1"
TAG="v${VERSION}"

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must match MAJOR.MINOR.PATCH (e.g., 0.2.2)."
    exit 1
fi

if [[ $# -eq 2 ]]; then
    if [[ "$2" != "--push" ]]; then
        echo "Error: unknown option '$2'."
        usage
        exit 1
    fi
    PUSH_TAG=1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is required but not available in PATH."
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: git worktree is not clean. Commit or stash changes first."
    exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: tag '$TAG' already exists."
    exit 1
fi

echo "Running checks in conda env '${ENV_NAME}'..."
conda run -n "$ENV_NAME" pytest -q
conda run -n "$ENV_NAME" ruff check src
conda run -n "$ENV_NAME" black --check src

echo "Creating annotated tag ${TAG}..."
git tag -a "$TAG" -m "Release ${TAG}"

if [[ "$PUSH_TAG" -eq 1 ]]; then
    echo "Pushing tag ${TAG} to origin..."
    git push origin "$TAG"
fi

echo "Done. Release tag created: ${TAG}"
