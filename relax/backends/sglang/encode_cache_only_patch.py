# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Monkey patch for SGLang's ``--encoder-only`` server: cache-only ``/encode``.

Why
---
Encoder/rollout overlap wants to run the vision tower for step ``N+1``'s images
while step ``N`` is still generating (and while the actor trains, when the
encoder is the only engine still resident).  The result should land in the
encoder's ``--enable-prefix-mm-cache`` embedding cache and nothing more: there
is no consumer yet, and the real request will arrive a whole step later.

SGLang's ``/encode`` cannot express that.  Under the default
``zmq_to_scheduler`` backend the handler always finishes with
``MMEncoder.send_with_url(req_id)``, which blocks on an ``asyncio.Condition``
until ``rid_to_receive_count[req_id]`` receivers have registered themselves via
``POST /scheduler_receive_url``.  A prefetch registers none, so ``expected_count``
stays ``None``, the loop never satisfies its break condition, and the request
hangs for ``SGLANG_ENCODER_SEND_TIMEOUT`` (180 s by default).  Registering a
throwaway sink instead would work but would push the embedding over TCP for no
reason -- 39-64 MiB per image, several GiB per step.

What this patch does
--------------------
Marks prefetch requests by ``req_id`` prefix (:data:`PREFETCH_RID_PREFIX`) and
short-circuits ``send_with_url`` for them.  By the time ``send_with_url`` is
reached the vision tower has already run and ``_encode`` has populated the
embedding cache (``encode_server.py``: cache fill happens inside ``_encode``,
before any send), so dropping the payload loses nothing.

Deliberately *not* patched: the ``/encode`` route handler itself.  It is bound
into the FastAPI app at import time, so replacing it would mean route surgery;
``send_with_url`` is the single choke point every zmq path funnels through.

The cache is keyed by image content hash, so a prefetch and the later real
request hit the same entry regardless of their differing ``req_id``.  Note this
is only sound because EPD requires ``--freeze-vision-model``: a trainable ViT
would make every cached embedding stale after the next weight sync, and the
encoder has no cache-invalidation route.
"""

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Prefetch requests carry this prefix so the encoder can recognise them without
# a schema change.  Kept deliberately distinct from SGLang's
# ``HEALTH_CHECK_RID_PREFIX``: that prefix disables metrics *and* the embedding
# cache (``log_metrics=False`` at encode_server.py, which gates
# ``use_mm_cache``), which would silently turn every prefetch into a no-op.
PREFETCH_RID_PREFIX = "relax-mmprefetch-"

_PATCH_FLAG = "_relax_encode_cache_only_patched"


def is_prefetch_request(req_id) -> bool:
    return isinstance(req_id, str) and req_id.startswith(PREFETCH_RID_PREFIX)


def apply_patch() -> bool:
    """Patch ``MMEncoder.send_with_url``.

    Idempotent; returns True if applied.
    """
    from sglang.srt.disaggregation.encode_server import MMEncoder

    if getattr(MMEncoder, _PATCH_FLAG, False):
        logger.info("[encode-cache-only] already applied, skip")
        return False

    _orig_send_with_url = MMEncoder.send_with_url

    async def _patched_send_with_url(self, req_id, *args, **kwargs):
        if not is_prefetch_request(req_id):
            return await _orig_send_with_url(self, req_id, *args, **kwargs)

        # Cache-only: the ViT forward is done and the embedding cache holds the
        # result.  Drop the payload rather than waiting out the send timeout for
        # receivers that will never register.
        embedding_to_send = getattr(self, "embedding_to_send", None)
        if embedding_to_send is not None:
            embedding_to_send.pop(req_id, None)
        logger.debug("[encode-cache-only] dropped embedding for prefetch req_id=%s", req_id)
        return None

    MMEncoder.send_with_url = _patched_send_with_url
    setattr(MMEncoder, _PATCH_FLAG, True)
    logger.info("[encode-cache-only] patched MMEncoder.send_with_url (prefix=%s)", PREFETCH_RID_PREFIX)
    return True


def apply_encode_cache_only_patch() -> None:
    """Apply the cache-only ``/encode`` patch inside an encoder server process.

    Never raises: a failed patch only costs the prefetch optimisation, and the
    encoder must still come up so the language engines can dispatch to it.
    """
    try:
        apply_patch()
    except Exception as e:  # never block encoder startup on the patch
        logger.warning("Failed to apply encode cache-only patch: %r", e)
