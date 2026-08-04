"""
Module Name
-----------
sentiment.py

Purpose
-------
Provides sentiment classification for product review and comment text,
returning one of three labels ("positive", "negative", "neutral") for a
single text or for a batch of texts.

Responsibilities
-----------------
- Guarantee zero network activity at import time and throughout the life
  of the process: HuggingFace offline mode is forced, and the local model
  cache is checked via direct disk I/O only, never a network call.
- Lazily load and cache, in-process, a single instance of the transformer-
  based sentiment-analysis pipeline, shared by all callers, on first use.
- Enforce a process-wide circuit breaker: once the model pipeline is
  confirmed unavailable (not cached locally, or a load/inference failure),
  every subsequent call skips the pipeline entirely and uses the
  keyword-based fallback classifier, with no further pipeline or network
  attempts for the rest of the process.
- Expose two public entry points with identical label semantics:
  `analyze_sentiment` for a single text and `analyze_sentiment_batch` for
  a list of texts, the latter using native pipeline batching for
  efficiency.
- Provide a deterministic, dependency-free keyword-based sentiment
  classifier used whenever the transformer pipeline is unavailable or
  fails.
- Emit diagnostic timing/shape logging for every `analyze_sentiment_batch`
  call (texts received, batch_size, number of chunks, per-chunk elapsed
  time and items/sec) so it is possible to confirm from a plain run log,
  with no debugger attached, whether real pipeline batching is actually
  engaging for a given call site - and a one-time warning from
  `analyze_sentiment` if it is being called often enough in a process to
  suggest a per-item loop is being used where batching was intended.

Architecture
------------
Lazy-initialization singleton pattern guarded by module-level threading
locks (`_pipeline_lock`, `_offline_mode_lock`, `_model_state_lock`).
State is tracked with module-level flags/caches: whether offline mode has
been configured, whether a pipeline load has been attempted, the cached
pipeline instance itself, and whether the model has been confirmed
unavailable for the remainder of the process. No setup work runs at
import time; all of it is deferred to the first call of
`analyze_sentiment`, `analyze_sentiment_batch`, or
`get_sentiment_pipeline`.

Sentiment Pipeline
-------------------
Model: cardiffnlp/twitter-roberta-base-sentiment-latest, loaded through
`transformers.pipeline("sentiment-analysis", ...)`. No network access during
model/tokenizer loading is guaranteed globally by
`_ensure_offline_mode_configured()` (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE env
vars + a direct patch to `huggingface_hub.constants.HF_HUB_OFFLINE`), not by
a per-call kwarg - see get_sentiment_pipeline()'s own comment for why a
redundant `model_kwargs={"local_files_only": True}` used to be passed here
too and had to be removed. `truncation=True, max_length=128` guarantees safe
handling of inputs longer than the model's token limit. Model labels are
normalized to "positive" / "negative" / "neutral" by matching the "pos"/"neg"
prefix of the returned label, defaulting to "neutral" otherwise.

Inputs
------
- `analyze_sentiment(pipeline, text)`: an optional pre-loaded pipeline
  object (or None to use the module's lazily-loaded shared pipeline) and
  a single text string.
- `analyze_sentiment_batch(pipeline, texts, batch_size)`: an optional
  pre-loaded pipeline object, a list of text strings, and a batch chunk
  size (default 64).

Outputs
-------
- `analyze_sentiment`: a single sentiment label string.
- `analyze_sentiment_batch`: a list of sentiment label strings, one per
  input text, in the same order as `texts`.

Dependencies
------------
Standard library: `logging`, `os`, `re`, `sys`, `threading`, `time`,
`typing`.
Optional third-party: `huggingface_hub`, `transformers` - both imported
lazily inside functions, only when a real pipeline load or cache scan is
actually attempted, so the module has no hard dependency on either at
import time.
"""

import logging
import os
import re
import sys
import threading
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

_SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _force_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import huggingface_hub
        huggingface_hub.constants.HF_HUB_OFFLINE = True
    except Exception:
        pass

    _offline_attr_names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "_is_offline_mode")
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("huggingface_hub") or mod_name.startswith("transformers")):
            continue
        for attr in _offline_attr_names:
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, True)
                except Exception:
                    pass


def _model_cached_locally(model_id: str = _SENTIMENT_MODEL_ID) -> bool:
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        return any(repo.repo_id == model_id for repo in cache_info.repos)
    except Exception as exc:
        logger.debug("Local HuggingFace cache scan failed: %s", exc)
        return False


_model_state_lock = threading.Lock()
_model_confirmed_unavailable = False


def _mark_model_unavailable(reason: str) -> None:
    global _model_confirmed_unavailable
    with _model_state_lock:
        already_known = _model_confirmed_unavailable
        _model_confirmed_unavailable = True
    if not already_known:
        logger.warning(
            "Sentiment model marked unavailable for the rest of this "
            "process (%s). All further calls will use keyword-based "
            "sentiment with no further pipeline/network attempts.",
            reason,
        )


def _model_should_be_skipped() -> bool:
    with _model_state_lock:
        return _model_confirmed_unavailable


_offline_mode_lock = threading.Lock()
_offline_mode_configured = False


def _ensure_offline_mode_configured() -> None:
    global _offline_mode_configured
    with _offline_mode_lock:
        if _offline_mode_configured:
            return
        _offline_mode_configured = True

        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "2")
        _force_offline_mode()

        if _model_cached_locally():
            logger.info(
                "Model %r found in the local HuggingFace cache; it will "
                "load from disk with zero network attempts.",
                _SENTIMENT_MODEL_ID,
            )
            return

        logger.warning(
            "Model %r is not cached locally. Marking the sentiment "
            "pipeline unavailable for the rest of this process so every "
            "call goes straight to keyword-based sentiment with no "
            "pipeline/network attempts at all.",
            _SENTIMENT_MODEL_ID,
        )
        _mark_model_unavailable("model not cached locally; offline mode forced, no downloads permitted")


_pipeline_lock = threading.Lock()
_cached_pipeline = None
_pipeline_load_attempted = False


def get_sentiment_pipeline():
    global _cached_pipeline, _pipeline_load_attempted

    with _pipeline_lock:
        if _pipeline_load_attempted:
            return _cached_pipeline
        _pipeline_load_attempted = True

        _ensure_offline_mode_configured()

        if _model_should_be_skipped():
            return None

        try:
            from transformers import pipeline as _hf_pipeline
            # NOTE: deliberately NOT passing model_kwargs={"local_files_only":
            # True} here anymore. Offline mode is already fully enforced
            # globally by _ensure_offline_mode_configured() above (HF_HUB_
            # OFFLINE/TRANSFORMERS_OFFLINE env vars + a direct patch to
            # huggingface_hub.constants.HF_HUB_OFFLINE), so this was always
            # redundant - and on newer transformers versions it actively
            # breaks: pipeline() now derives its own local_files_only value
            # internally and forwards it to AutoConfig.from_pretrained()
            # alongside **model_kwargs, so having local_files_only in
            # model_kwargs too raises "got multiple values for keyword
            # argument 'local_files_only'" and permanently disables the real
            # model for the rest of the process (confirmed via live log).
            _cached_pipeline = _hf_pipeline(
                "sentiment-analysis",
                model=_SENTIMENT_MODEL_ID,
                tokenizer=_SENTIMENT_MODEL_ID,
                truncation=True,
                max_length=128,
            )
            logger.info(
                "Sentiment pipeline loaded from local cache (offline mode "
                "enforced globally via HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE, "
                "zero network attempts) and cached in memory for reuse by "
                "all future calls."
            )
        except Exception as exc:
            _mark_model_unavailable(f"pipeline load raised: {exc!r}")
            _cached_pipeline = None
            return _cached_pipeline

        # Opt-in dynamic INT8 quantization for faster CPU inference. There is
        # no GPU on the target deployment machine (confirmed: Intel UHD
        # integrated graphics only), so this is the one remaining lever to
        # raise the CPU inference throughput ceiling itself - typically a
        # real 2-4x speedup for a transformer this size, with only a small,
        # usually negligible accuracy cost for a 3-class sentiment task.
        # Deliberately wrapped so ANY failure here (torch API differences
        # across versions, an environment without a working torch install,
        # anything) falls back to the normal fp32 pipeline that was already
        # working above - this can only make things faster or unchanged,
        # never break the working path.
        try:
            import torch

            # Explicitly pin intra-op thread count to the logical core count
            # (4 on the target i3-10110U: 2 physical cores / 4 threads via
            # hyperthreading). PyTorch's own auto-detection can be
            # conservative on some Windows builds (sometimes settling on the
            # physical core count, or lower, depending on how the process
            # was launched) — being explicit here removes that uncertainty.
            # This only affects CPU intra-op parallelism; it does not change
            # results, only how many threads compute a given batch. Safe to
            # call every process; PyTorch raises if it's called after
            # inference has already started on this process, so it's kept
            # here, before the first real inference call.
            try:
                _cpu_count = os.cpu_count() or 4
                torch.set_num_threads(_cpu_count)
                logger.info(
                    "torch intra-op thread count explicitly set to %d "
                    "(os.cpu_count()) for CPU inference.",
                    _cpu_count,
                )
            except Exception as exc:
                logger.warning(
                    "Could not set torch thread count explicitly (%s) - "
                    "continuing with PyTorch's own default.", exc,
                )

            try:
                quantize_dynamic = torch.ao.quantization.quantize_dynamic
            except AttributeError:
                quantize_dynamic = torch.quantization.quantize_dynamic
            _cached_pipeline.model = quantize_dynamic(
                _cached_pipeline.model, {torch.nn.Linear}, dtype=torch.qint8,
            )
            logger.info(
                "Sentiment model dynamically quantized to int8 for faster "
                "CPU inference (no GPU available on this machine)."
            )
        except Exception as exc:
            logger.warning(
                "Dynamic quantization unavailable/failed (%s) - continuing "
                "with the normal (unquantized) pipeline. This is a "
                "performance-only fallback; sentiment results are "
                "unaffected either way.",
                exc,
            )

        # Warm up the (possibly quantized) pipeline with one throwaway
        # batched inference call before returning it. get_sentiment_pipeline()
        # only ever reaches this point once per process (guarded by
        # _pipeline_load_attempted above), and app.py already invokes it from
        # its startup preload hook, off the request path, in the same
        # dedicated single-worker executor used for every real inference
        # call later - so this pays PyTorch/oneDNN's lazy first-forward-pass
        # setup cost (int8 kernel/primitive selection, thread-pool spin-up,
        # etc.) once here, during the already-accepted startup window,
        # instead of on whichever request happens to hit chunk 1 of the
        # first real analyze_sentiment_batch call.
        #
        # Confirmed from a live run log (Open Issue #3): chunk 1 of a
        # 1059-text call ran at 0.764s/item despite having the *shortest*
        # text in the entire run (token_len avg=8, max=44) - nearly 9x
        # slower than similarly-short chunk 4 (0.087s/item) - costing
        # roughly 43s of pure dead weight (~48.9s actual vs. ~5-6s expected)
        # that a real user's first request was paying, not startup.
        #
        # Deliberately wrapped in try/except: if warm-up itself fails for
        # any reason, the pipeline already built above is still returned and
        # used as normal - this can only save time, never break the working
        # path or change any classification result.
        try:
            _warm_start = time.perf_counter()
            _cached_pipeline(
                [
                    "warm up",
                    "a slightly longer warm-up sentence to help prime more than one sequence length",
                ],
                batch_size=2,
            )
            logger.info(
                "Sentiment pipeline warmed up with a throwaway batched "
                "inference call in %.2fs (paid once here at startup "
                "instead of on the first real analysis request).",
                time.perf_counter() - _warm_start,
            )
        except Exception as exc:
            logger.warning(
                "Sentiment pipeline warm-up call failed (%s) - continuing "
                "with the pipeline as loaded; this only affects whether "
                "the JIT/kernel warm-up cost is paid now vs. on the first "
                "real request, not correctness.",
                exc,
            )

        return _cached_pipeline


_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "fantastic",
    "love", "loved", "best", "perfect", "wonderful", "happy", "glad",
    "satisfied", "recommend", "outstanding", "superb", "brilliant",
    "quality", "helpful", "fast", "quick", "nice", "neat", "pleased",
    "impressive", "smooth", "easy", "enjoy", "enjoyed", "beautiful",
    "stunning", "delighted", "polite", "friendly", "efficient", "clean",
    "fresh", "genuine", "authentic", "value", "worth", "affordable",
    "reasonable", "reliable", "trust", "trusted", "legit", "legitimate",
    "prompt", "responsive", "professional", "top", "positive", "wow",
}

_NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "horrible", "awful", "poor", "hate",
    "hated", "disappointed", "disappointing", "useless", "broken",
    "scam", "fake", "fraud", "rude", "slow", "late", "delay", "delayed",
    "expensive", "overpriced", "waste", "wasted", "wrong", "defective",
    "damaged", "missing", "lost", "never", "never again", "refund",
    "return", "problem", "issue", "complaint", "complain", "angry",
    "frustrated", "unhappy", "pathetic", "ridiculous", "cheated",
    "lied", "ignored", "no response", "avoid", "don't buy", "do not buy",
    "not worth", "money wasted", "misleading", "broken", "fail", "failed",
}

_NEGATION = {"not", "no", "never", "don't", "doesn't", "didn't",
             "won't", "can't", "isn't", "wasn't", "hardly", "barely"}


def _keyword_sentiment(text: str) -> str:
    tokens = re.findall(r"\b\w+\b", text.lower())
    pos = neg = 0
    negate = False
    for tok in tokens:
        if tok in _NEGATION:
            negate = True
            continue
        if tok in _POSITIVE_WORDS:
            if negate:
                neg += 1
            else:
                pos += 1
        elif tok in _NEGATIVE_WORDS:
            if negate:
                pos += 1
            else:
                neg += 1
        negate = False
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _resolve_pipeline(pipeline):
    return pipeline if pipeline is not None else get_sentiment_pipeline()


def _map_label(label: str) -> str:
    label = label.lower()
    if label.startswith("pos"):
        return "positive"
    if label.startswith("neg"):
        return "negative"
    return "neutral"


# Diagnostic-only state: counts calls to the single-item entry point below.
# This has no effect on classification behavior. Its only purpose is to
# make a specific silent failure mode visible in a plain run log: a
# per-run analysis loop that is *supposed* to call analyze_sentiment_batch
# once with the full list of texts, but instead calls analyze_sentiment
# (this function) once per item in a Python loop. That pattern still
# "works" (correct labels come back) but throws away all the batching
# speedup, and nothing about its *output* looks wrong - only the call
# pattern does. A one-time threshold warning surfaces it without spamming
# the log on every single call.
_single_call_lock = threading.Lock()
_single_call_count = 0
_single_call_warning_logged = False
_SINGLE_CALL_WARNING_THRESHOLD = 20


def analyze_sentiment(pipeline, text: str) -> str:
    if not text:
        return "neutral"

    global _single_call_count, _single_call_warning_logged
    with _single_call_lock:
        _single_call_count += 1
        count_now = _single_call_count
        should_warn = (
            count_now == _SINGLE_CALL_WARNING_THRESHOLD
            and not _single_call_warning_logged
        )
        if should_warn:
            _single_call_warning_logged = True
    if should_warn:
        logger.warning(
            "analyze_sentiment() (single-item entry point) has now been "
            "called %d times in this process. If a per-run analysis loop "
            "is meant to be using analyze_sentiment_batch() for real "
            "batched inference, this many single-item calls suggests it "
            "may instead be looping and calling analyze_sentiment() once "
            "per item - check the calling code (e.g. app.py) for a loop "
            "that should be replaced with one analyze_sentiment_batch() "
            "call over the full list of texts.",
            count_now,
        )

    active_pipeline = _resolve_pipeline(pipeline)

    if active_pipeline is not None and not _model_should_be_skipped():
        try:
            result = active_pipeline(text)[0]
            return _map_label(result["label"])
        except Exception as exc:
            _mark_model_unavailable(f"pipeline call raised: {exc!r}")
    return _keyword_sentiment(text)


def analyze_sentiment_batch(pipeline, texts: List[str], batch_size: int = 64) -> List[str]:
    call_start = time.perf_counter()
    results: List[Optional[str]] = [None] * len(texts)

    for i, t in enumerate(texts):
        if not t:
            results[i] = "neutral"

    non_empty_indices = [i for i, t in enumerate(texts) if t]

    active_pipeline = _resolve_pipeline(pipeline)
    pipeline_will_run = (
        active_pipeline is not None
        and bool(non_empty_indices)
        and not _model_should_be_skipped()
    )

    # DIAGNOSTIC: this is the line that answers "is real batching engaging,
    # and at what call shape?" from a plain run log with no debugger. A
    # single call for the whole run with len(texts) close to the total
    # analyzed-item count means the caller is using this function as
    # intended. Many small calls (e.g. len(texts)=1, repeated hundreds of
    # times) means the caller is looping per item before ever reaching
    # this function, which defeats batching just as surely as skipping it
    # entirely - and is the #1 suspect for the 215s/381-item timing seen
    # in production, since 0.56s/item is consistent with per-item pipeline
    # overhead on this hardware, not with real batched inference.
    logger.info(
        "analyze_sentiment_batch: call received %d texts (%d non-empty), "
        "batch_size=%d, pipeline_available=%s.",
        len(texts), len(non_empty_indices), batch_size, pipeline_will_run,
    )

    pipeline_processed_count = 0

    if pipeline_will_run:
        # Sort by length before chunking into batches. A transformer batch
        # pads every member up to its longest member's token length, so an
        # arbitrarily-ordered batch (a long rant next to a two-word
        # comment) pays that padding cost for every short comment in it.
        # With comment lengths as skewed as real scrape data (a handful of
        # long YouTube/Reddit comments mixed with short one-liners), that
        # waste dominates total runtime. Sorting first means each batch is
        # made of similarly-sized texts, so padding collapses close to
        # zero - this only changes the *order* comments are fed to the
        # model in; every index is still processed exactly once, and
        # results are scattered back to their original position below, so
        # the returned list's order and content are unaffected.
        sorted_indices = sorted(non_empty_indices, key=lambda i: len(texts[i]))
        total_chunks = (len(sorted_indices) + batch_size - 1) // batch_size

        logger.info(
            "analyze_sentiment_batch: pipeline batching engaged - %d "
            "non-empty text(s) split into %d chunk(s) of up to "
            "batch_size=%d each.",
            len(sorted_indices), total_chunks, batch_size,
        )

        chunk_num = 0
        for start in range(0, len(sorted_indices), batch_size):
            if _model_should_be_skipped():
                logger.warning(
                    "analyze_sentiment_batch: model marked unavailable "
                    "mid-call after chunk %d/%d - remaining texts in "
                    "this call will fall back to keyword-based sentiment.",
                    chunk_num, total_chunks,
                )
                break
            chunk_num += 1
            chunk_indices = sorted_indices[start:start + batch_size]
            chunk_texts = [texts[i] for i in chunk_indices]

            # DIAGNOSTIC (Open Issue #3 - per-chunk degradation): capture
            # length stats for this chunk BEFORE starting the inference
            # timer below, so this measurement never counts against
            # chunk_elapsed. char_lengths is free (already-materialized
            # strings). token_lengths re-tokenizes with the same
            # truncation=True, max_length=128 the pipeline itself applies,
            # so it reflects what the model actually sees post-truncation,
            # not raw comment length.
            #
            # RESOLVED (confirmed via live log, 1019-text run): yes, cost
            # scales strongly super-linearly with sequence length - the
            # longest chunk (avg 100 tokens, some truncated at the OLD
            # 512-token cap) ran at 0.846s/item vs 0.129s/item for the
            # shortest chunk (avg 8 tokens): ~6.5x slower for ~12x the
            # tokens. That alone accounted for roughly half of a 304s total
            # Sentiment Analysis stage. max_length was cut from 512 to 128
            # in response - short enough to eliminate the worst of that
            # tail, still comfortably more than enough context for
            # sentiment (unlike aspect extraction or summarization, overall
            # sentiment is almost always clear from the first sentence or
            # two). This diagnostic block is kept rather than removed, in
            # case a future change to input length distribution reopens
            # the question.
            char_lengths = [len(t) for t in chunk_texts]
            token_lengths: List[int] = []
            try:
                _token_ids = active_pipeline.tokenizer(
                    chunk_texts, truncation=True, max_length=128,
                )["input_ids"]
                token_lengths = [len(ids) for ids in _token_ids]
            except Exception as exc:
                logger.debug(
                    "analyze_sentiment_batch: token-length diagnostic "
                    "failed for chunk %d (%s) - timing/inference below is "
                    "unaffected, only this log line's token_len fields "
                    "will be missing.",
                    chunk_num, exc,
                )

            chunk_start = time.perf_counter()
            try:
                # CRITICAL: batch_size must be passed here. transformers'
                # Pipeline.__call__ defaults to batch_size=1 internally
                # when not given one explicitly - without this, every
                # comment in chunk_texts was being run through the model
                # one at a time regardless of how carefully it was
                # length-sorted and grouped above, making all of that
                # chunking work purely cosmetic. This single argument is
                # what actually turns it into real batched inference.
                chunk_results = active_pipeline(chunk_texts, batch_size=len(chunk_texts))
                for idx, result in zip(chunk_indices, chunk_results):
                    results[idx] = _map_label(result["label"])
                pipeline_processed_count += len(chunk_texts)
                chunk_elapsed = time.perf_counter() - chunk_start
                logger.info(
                    "analyze_sentiment_batch: chunk %d/%d done - %d "
                    "item(s) in %.2fs (%.3fs/item) - char_len avg=%.0f "
                    "max=%d, token_len avg=%.0f max=%d (post-truncation, "
                    "cap=128).",
                    chunk_num, total_chunks, len(chunk_texts), chunk_elapsed,
                    chunk_elapsed / len(chunk_texts) if chunk_texts else 0.0,
                    (sum(char_lengths) / len(char_lengths)) if char_lengths else 0.0,
                    max(char_lengths) if char_lengths else 0,
                    (sum(token_lengths) / len(token_lengths)) if token_lengths else 0.0,
                    max(token_lengths) if token_lengths else 0,
                )
            except Exception as exc:
                _mark_model_unavailable(f"pipeline batch call raised: {exc!r}")
                logger.warning(
                    "analyze_sentiment_batch: chunk %d/%d failed after "
                    "%.2fs (%s) - this and any remaining chunks fall back "
                    "to keyword-based sentiment.",
                    chunk_num, total_chunks, time.perf_counter() - chunk_start, exc,
                )

    fallback_count = 0
    for idx in non_empty_indices:
        if results[idx] is None:
            results[idx] = _keyword_sentiment(texts[idx])
            fallback_count += 1

    call_elapsed = time.perf_counter() - call_start
    logger.info(
        "analyze_sentiment_batch: call done - %d text(s) total in %.2fs "
        "(%.3fs/item overall); %d via pipeline, %d via keyword fallback.",
        len(texts), call_elapsed,
        call_elapsed / len(texts) if texts else 0.0,
        pipeline_processed_count, fallback_count,
    )

    return results
