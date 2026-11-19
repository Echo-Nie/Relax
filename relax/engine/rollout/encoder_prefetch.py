# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cross-step encoder prefetch for encoder-disaggregated (EPD) rollout.

Idea
----
Under ``--colocate`` the dedicated encoder GPU is idle for most of a step: it
finishes its share of the work early in the rollout phase, then sits out the
teacher tail *and* the whole training phase (it is the one engine that is never
offloaded, because SGLang's ``--encoder-only`` server has no memory saver).
Meanwhile ``generate_rollout`` already submits step ``N+1``'s
``data_buffer.get_samples()`` at the end of step ``N``.

So: take that same ObjectRef, resolve it in the background, and push step
``N+1``'s images through the encoder while step ``N`` is still finishing.  When
step ``N+1`` starts, the language engines dispatch to the encoder as usual and
hit a warm embedding cache instead of paying for the vision tower inline.

This warms a cache and nothing else.  It never mutates samples, so a failure —
encoder down, request timeout, an unexpected image type — costs performance and
only performance: the language engine recomputes the embedding locally exactly
as it does today.

Correctness
-----------
The encoder's embedding cache is keyed by image content hash with no
weight-version component, and embeddings computed here are necessarily produced
*before* the step-N weight sync.  Both are only sound because encoder
disaggregation requires ``--freeze-vision-model`` (enforced in
``relax/backends/sglang/arguments.py``); with a trainable vision tower these
embeddings would be one optimizer step stale.

The second ``ray.get`` on the prefetched ref is deliberate: the rollout loop
resolves the very same ref again at the start of the next step.  Ray serves the
object from the local store the second time, so this costs a deserialisation,
not a fetch — and it keeps the consumption path completely unchanged.
"""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from time import monotonic
from typing import Any

import ray
import requests

from relax.backends.sglang.encode_cache_only_patch import PREFETCH_RID_PREFIX
from relax.utils.data.processing_utils import encode_image_for_rollout_engine
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def get_encoder_urls(args) -> list[str]:
    """Encoder endpoints published by ``start_rollout_servers``, if any."""
    return list(getattr(args, "rollout_encoder_urls", None) or [])


def is_encoder_prefetch_enabled(args) -> bool:
    return bool(get_encoder_urls(args)) and Envs.RELAX_ENCODER_PREFETCH


def _iter_group_images(groups: Any):
    """Yield the PIL images of each group, once per group.

    ``get_samples`` returns groups of ``n_samples_per_prompt`` samples that
    share one ``multimodal_inputs`` dict by reference (``_shallow_copy_sample``
    in ``data_source.py``), so encoding the first sample of each group covers
    the whole group without duplicating work.
    """
    for group in groups or []:
        sample = group[0] if isinstance(group, list) and group else group
        if isinstance(sample, list) and sample:  # nested (agentic) shape
            sample = sample[0]
        mm_inputs = getattr(sample, "multimodal_inputs", None)
        if not mm_inputs:
            continue
        images = mm_inputs.get("images")
        if images:
            yield images


class EncoderPrefetcher:
    """Warms the encoder embedding cache with the next step's images.

    One driver thread per step; a still-running driver from the previous step
    is left alone and the new request is skipped, so a slow encoder degrades to
    "no prefetch" rather than piling up threads.
    """

    def __init__(self, args) -> None:
        self.args = args
        self._driver: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, samples_ref) -> None:
        """Kick off a prefetch for *samples_ref*.

        Never blocks, never raises.
        """
        try:
            if samples_ref is None or not is_encoder_prefetch_enabled(self.args):
                return
            with self._lock:
                if self._driver is not None and self._driver.is_alive():
                    logger.warning(
                        "Encoder prefetch for the previous step is still running; skipping this step. "
                        "The encoder is the bottleneck if this repeats."
                    )
                    return
                self._driver = threading.Thread(
                    target=self._run,
                    args=(samples_ref,),
                    name="encoder-prefetch",
                    daemon=True,
                )
                self._driver.start()
        except Exception as e:  # a prefetch must never break the rollout
            logger.warning(f"Failed to start encoder prefetch: {e!r}")

    def _run(self, samples_ref) -> None:
        started = monotonic()
        urls = get_encoder_urls(self.args)
        try:
            groups = ray.get(samples_ref)
        except Exception as e:
            logger.warning(f"Encoder prefetch could not resolve the next-step samples: {e!r}")
            return

        try:
            payloads: list[str] = []
            for images in _iter_group_images(groups):
                payloads.extend(encode_image_for_rollout_engine(image) for image in images)
        except Exception as e:
            logger.warning(f"Encoder prefetch failed while encoding images: {e!r}")
            return

        if not payloads:
            return

        concurrency = max(1, Envs.RELAX_ENCODER_PREFETCH_CONCURRENCY)
        timeout_s = Envs.RELAX_ENCODER_PREFETCH_TIMEOUT_S
        with requests.Session() as session, ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = pool.map(
                lambda item: self._encode_one(session, urls[item[0] % len(urls)], item[1], timeout_s),
                enumerate(payloads),
            )
            ok = sum(1 for r in results if r)

        elapsed = monotonic() - started
        log = logger.info if ok == len(payloads) else logger.warning
        log(f"Encoder prefetch: warmed {ok}/{len(payloads)} images in {elapsed:.1f}s across {len(urls)} encoder(s)")

    def _encode_one(self, session: requests.Session, url: str, image_b64: str, timeout_s: int) -> bool:
        # Mirrors the payload MMReceiverBase._send_encode_request builds, with
        # the Relax prefetch rid prefix so the patched encoder skips the send
        # instead of blocking on receivers that will never register.
        payload = {
            "req_id": f"{PREFETCH_RID_PREFIX}{uuid.uuid4().hex}",
            "modality": "IMAGE",
            "mm_items": [image_b64],
            "num_parts": 1,
            "part_idx": 0,
            "prefill_host": "",
            "embedding_port": None,
        }
        try:
            resp = session.post(f"{url.rstrip('/')}/encode", json=payload, timeout=timeout_s)
            if resp.status_code != 200:
                logger.warning(f"Encoder prefetch got HTTP {resp.status_code} from {url}: {resp.text[:200]}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Encoder prefetch request to {url} failed: {e!r}")
            return False
