# ADR 017: Self-Hosted Worker Canonicalization

## Status

Accepted

## Context

The self-hosted track currently describes workers in several incompatible ways:

1. `docs/deployment/SELF_HOSTED.md` presents the current production path around `scripts/install.sh`, `deploy/docker-compose.production.yml`, and an optional `workers` compose profile, but it names the runtime only as `debate-worker`.
2. `docs/deployment/SELF_HOSTED_GUIDE.md` still describes legacy root-level compose files such as `docker-compose.simple.yml`, `docker-compose.sme.yml`, and `docker-compose.production.yml`.
3. Health and verification examples drift between `/api/health` and `/api/v1/health`, which means the deploy docs do not point to one stable self-host verification surface.
4. The repository exposes more than one worker implementation:
   - `python -m scripts.queue_worker` processes debate jobs from Redis Streams.
   - `python -m scripts.control_plane_deliberation_worker` claims control-plane `deliberation` tasks.
   - `docs/deployment/ASYNC_GATEWAY.md` discusses ASGI or Gunicorn worker processes, which are HTTP serving choices rather than queue-processing workers.

Without an ADR, later documentation edits can continue to mix deployment topology, queue execution, and web-server process models under the same generic "worker" term.

## Decision

For the self-hosted track, Aragora documents one canonical background worker invocation:

```bash
python -m scripts.queue_worker --worker-id <worker-id> --concurrency <n>
```

This ADR standardizes the interpretation of worker paths as follows:

1. **Canonical self-hosted worker path**
   `scripts.queue_worker` is the baseline queue-processing worker for self-hosted debate execution. Any self-hosted guide that refers to "worker containers", "debate workers", or worker scale-out should treat this command as the runtime contract inside the container or host process.
2. **Canonical deployment anchor**
   Self-hosted deployment documentation should anchor on `scripts/install.sh` and `deploy/docker-compose.production.yml` as the current production compose path. Legacy root-level compose filenames are documentation drift unless and until they are restored as supported artifacts.
3. **Deploy-config drift must be called out explicitly**
   Documentation that still references `docker-compose.simple.yml`, `docker-compose.sme.yml`, `docker-compose.production.yml` at the repository root, or `/api/health` as the primary self-host verification endpoint is drift relative to the current self-hosted production path. That drift should be removed or labeled as legacy, not silently repeated.
4. **Alternate worker paths are classified, not merged**
   `scripts.control_plane_deliberation_worker` is a specialized control-plane worker for claimed `deliberation` tasks and is not the default self-hosted debate worker.
   ASGI or Gunicorn worker counts in `docs/deployment/ASYNC_GATEWAY.md` are web-serving concurrency controls and are not substitutes for `scripts.queue_worker`.
   Compose service names such as `debate-worker` or `aragora-worker` are packaging labels only; they should resolve back to an explicit underlying invocation instead of becoming the contract themselves.

## Consequences

### Positive

- Self-hosted docs can name one concrete worker command instead of relying on ambiguous service labels.
- Reviewers get a clear standard for spotting drift between deployment docs, validation scripts, and runtime expectations.
- Specialized workers remain available without confusing the default self-hosted debate-processing story.

### Negative

- Existing deployment docs still contain drift and will need follow-up edits outside this ADR lane.
- Teams using specialized workers must document those paths explicitly instead of relying on generic "worker" wording.

### Follow-Up Implications

- Future self-hosted doc updates should reconcile `SELF_HOSTED.md`, `SELF_HOSTED_GUIDE.md`, and related deployment runbooks against this ADR.
- If the production compose stack later adopts a different worker entrypoint, that change should update the implementation and supersede this ADR rather than adding another competing term.
