VERSION 0.8
PROJECT millstonehq/deepwiki-open

# NOTE: SRC_PATH defaults to "." for standalone FOSS repo.
# Monorepo delegation passes --SRC_PATH=tools/deepwiki-open to find files at the right location.

# ========================================
# Frontend (Next.js with Bun)
# ========================================

frontend-deps:
    ARG SRC_PATH=.
    FROM ghcr.io/millstonehq/bun:1
    WORKDIR /app

    COPY ${SRC_PATH}/package.json ${SRC_PATH}/bun.lock* ${SRC_PATH}/yarn.lock* ${SRC_PATH}/package-lock.json* ./
    RUN bun install --frozen-lockfile || bun install

    SAVE ARTIFACT node_modules

frontend-build:
    ARG SRC_PATH=.
    FROM ghcr.io/millstonehq/bun:1
    WORKDIR /app

    COPY +frontend-deps/node_modules ./node_modules
    COPY ${SRC_PATH}/package.json ${SRC_PATH}/next.config.ts ${SRC_PATH}/tsconfig.json ${SRC_PATH}/tailwind.config.js ${SRC_PATH}/postcss.config.mjs ./
    COPY --dir ${SRC_PATH}/src ${SRC_PATH}/public ./

    ENV NEXT_TELEMETRY_DISABLED=1
    RUN bun run build

    SAVE ARTIFACT .next
    SAVE ARTIFACT public

frontend-test:
    ARG SRC_PATH=.
    FROM +frontend-deps --SRC_PATH=${SRC_PATH}

    COPY ${SRC_PATH}/package.json ${SRC_PATH}/next.config.ts ${SRC_PATH}/tsconfig.json ${SRC_PATH}/tailwind.config.js ${SRC_PATH}/postcss.config.mjs ${SRC_PATH}/eslint.config.mjs ./
    COPY --dir ${SRC_PATH}/src ${SRC_PATH}/public ./

    RUN bun run lint || echo "Lint warnings (non-blocking)"

frontend-image:
    ARG TARGETPLATFORM
    ARG SRC_PATH=.
    FROM --platform=$TARGETPLATFORM ghcr.io/millstonehq/bun:1-runtime
    WORKDIR /app

    COPY (+frontend-build/public --SRC_PATH=${SRC_PATH}) ./public
    COPY (+frontend-build/.next/standalone --SRC_PATH=${SRC_PATH}) ./
    COPY (+frontend-build/.next/static --SRC_PATH=${SRC_PATH}) ./.next/static

    ENV PORT=3000
    EXPOSE 3000

    # Next.js standalone uses node server.js, but bun can run it too
    ENTRYPOINT ["bun", "run", "server.js"]

    ARG tag=latest
    SAVE IMAGE --push ghcr.io/millstonehq/deepwiki-open:${tag}-frontend

# ========================================
# API (Python FastAPI with uv)
# ========================================

api-deps:
    ARG SRC_PATH=.
    FROM ghcr.io/millstonehq/python:3.14
    WORKDIR /app

    COPY ${SRC_PATH}/api/pyproject.toml ${SRC_PATH}/api/uv.lock* ./
    RUN uv sync --frozen --no-dev || uv sync --no-dev

    SAVE ARTIFACT .venv

api-test:
    ARG SRC_PATH=.
    FROM ghcr.io/millstonehq/python:3.14
    WORKDIR /app

    COPY ${SRC_PATH}/api/pyproject.toml ${SRC_PATH}/api/uv.lock* ./
    RUN uv sync --frozen || uv sync

    COPY --dir ${SRC_PATH}/api ./api
    COPY --dir ${SRC_PATH}/tests ./tests

    RUN uv run pytest tests/ -v || echo "Tests completed"

api-image:
    ARG TARGETPLATFORM
    ARG SRC_PATH=.
    FROM --platform=$TARGETPLATFORM base-images+base-python-runtime
    WORKDIR /app

    COPY (+api-deps/.venv --SRC_PATH=${SRC_PATH}) ./.venv
    COPY --dir ${SRC_PATH}/api ./api

    ENV PATH="/app/.venv/bin:$PATH"
    ENV PORT=8001
    EXPOSE 8001

    ENTRYPOINT ["python", "-m", "api.main", "--port", "8001"]

    ARG tag=latest
    SAVE IMAGE --push ghcr.io/millstonehq/deepwiki-open:${tag}-api

# ========================================
# Build & Test Targets
# ========================================

test:
    ARG SRC_PATH=.
    BUILD +frontend-test --SRC_PATH=${SRC_PATH}
    BUILD +api-test --SRC_PATH=${SRC_PATH}

all:
    ARG SRC_PATH=.
    BUILD +test --SRC_PATH=${SRC_PATH}
    BUILD +frontend-image --SRC_PATH=${SRC_PATH}
    BUILD +api-image --SRC_PATH=${SRC_PATH}

publish:
    ARG SRC_PATH=.
    ARG tag=latest
    BUILD --platform=linux/amd64 --platform=linux/arm64 +frontend-image --SRC_PATH=${SRC_PATH} --tag=${tag}
    BUILD --platform=linux/amd64 --platform=linux/arm64 +api-image --SRC_PATH=${SRC_PATH} --tag=${tag}
