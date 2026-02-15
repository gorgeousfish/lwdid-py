"""
Warning registry for deferred collection, aggregation, and output.

Implements a collect-aggregate-flush pattern for warnings generated
during (cohort, period) iteration loops in staggered adoption estimation.
Instead of emitting warnings immediately (which produces hundreds of
duplicates), the registry buffers them and emits aggregated summaries
after the loop completes.

Three verbosity levels control output behavior:

- ``quiet``   : Emit only critical warnings (convergence, numerical).
- ``default`` : Emit one aggregated summary per warning category.
- ``verbose`` : Emit every individual warning record.

The ``get_diagnostics()`` method always returns the full record set
regardless of verbosity, so users can inspect all warnings post-hoc
via ``LWDIDResults.diagnostics``.
"""

import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .warnings_categories import (
    ConvergenceWarning,
    NumericalWarning,
)

_VALID_VERBOSE_LEVELS = frozenset({'quiet', 'default', 'verbose'})

# Categories treated as critical: emitted even in quiet mode.
_CRITICAL_CATEGORIES = frozenset({ConvergenceWarning, NumericalWarning})


@dataclass
class WarningRecord:
    """Single warning record captured by the registry."""

    category: type
    message: str
    cohort: Any = None
    period: Any = None
    context: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class WarningRegistry:
    """
    Centralized warning collector for staggered adoption estimation.

    Buffers warnings during (g, r) iteration and emits aggregated
    summaries (or individual records) when ``flush()`` is called.
    Diagnostics are always available via ``get_diagnostics()``
    regardless of verbosity setting.

    Parameters
    ----------
    verbose : str, default ``'default'``
        Output verbosity level. One of ``'quiet'``, ``'default'``,
        ``'verbose'``. Case-insensitive.

    Raises
    ------
    ValueError
        If *verbose* is not one of the three valid levels.
    """

    def __init__(self, verbose: str = 'default') -> None:
        normalized = verbose.lower() if isinstance(verbose, str) else verbose
        if normalized not in _VALID_VERBOSE_LEVELS:
            raise ValueError(
                f"Invalid verbose level {verbose!r}. "
                f"Must be one of {sorted(_VALID_VERBOSE_LEVELS)}."
            )
        self._verbose: str = normalized
        self._records: list[WarningRecord] = []
        self._flushed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        category: type,
        message: str,
        cohort: Any = None,
        period: Any = None,
        context: dict | None = None,
    ) -> None:
        """
        Buffer a warning record without emitting it.

        Parameters
        ----------
        category : type
            Warning class (subclass of :class:`LWDIDWarning`).
        message : str
            Human-readable warning text.
        cohort : int or None
            Treatment cohort identifier, if applicable.
        period : int or None
            Calendar period identifier, if applicable.
        context : dict or None
            Auxiliary numeric context (e.g. ``{'n_treated': 3}``).
        """
        self._records.append(
            WarningRecord(
                category=category,
                message=message,
                cohort=cohort,
                period=period,
                context=context if context is not None else {},
            )
        )

    def flush(self, total_pairs: int | None = None) -> None:
        """
        Aggregate and emit buffered warnings, then mark as flushed.

        Subsequent calls are no-ops (idempotent). An empty registry
        produces no output and no exception.

        Parameters
        ----------
        total_pairs : int or None
            Total number of (cohort, period) pairs processed. Used
            in aggregated summary messages to report M/T ratios.
        """
        if self._flushed:
            return
        self._flushed = True

        if not self._records:
            return

        if self._verbose == 'quiet':
            self._flush_quiet(total_pairs)
        elif self._verbose == 'default':
            self._flush_default(total_pairs)
        else:  # verbose
            self._flush_verbose()

    def get_diagnostics(self) -> list[dict]:
        """
        Return structured diagnostic data for all collected warnings.

        The output is independent of the verbosity setting and always
        contains the full record set, aggregated by category.

        Returns
        -------
        list of dict
            Each dict contains:

            - ``category`` : str — warning class name.
            - ``message`` : str — representative message text.
            - ``count`` : int — number of occurrences.
            - ``affected_pairs`` : list of (cohort, period) tuples.
            - ``context_summary`` : dict — min/max of numeric context fields.
        """
        if not self._records:
            return []

        grouped = self._aggregate_by_category()
        diagnostics = []
        for cat, records in grouped.items():
            affected = [
                (r.cohort, r.period)
                for r in records
                if r.cohort is not None or r.period is not None
            ]
            diagnostics.append({
                'category': cat.__name__,
                'message': records[0].message,
                'count': len(records),
                'affected_pairs': affected,
                'context_summary': self._summarize_context(records),
            })
        return diagnostics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_by_category(self) -> dict[type, list[WarningRecord]]:
        """Group records by warning category, preserving insertion order."""
        grouped: dict[type, list[WarningRecord]] = defaultdict(list)
        for rec in self._records:
            grouped[rec.category].append(rec)
        return dict(grouped)

    def _format_summary(
        self,
        category: type,
        records: list[WarningRecord],
        total_pairs: int | None,
    ) -> str:
        """
        Build an aggregated summary string for one category.

        Includes the count of affected pairs and the M/T ratio when
        *total_pairs* is provided.
        """
        count = len(records)
        cat_name = category.__name__

        # Collect affected (g, r) pairs for the ratio.
        affected = [
            (r.cohort, r.period)
            for r in records
            if r.cohort is not None or r.period is not None
        ]
        n_affected = len(affected)

        if total_pairs is not None and total_pairs > 0 and n_affected > 0:
            ratio_str = f"{n_affected}/{total_pairs} (cohort, period) pairs"
        elif n_affected > 0:
            ratio_str = f"{n_affected} (cohort, period) pairs"
        else:
            ratio_str = f"{count} occurrences"

        # Use the first record's message as representative text.
        representative = records[0].message

        return (
            f"[{cat_name}] {representative} "
            f"({ratio_str})"
        )

    @staticmethod
    def _summarize_context(records: list[WarningRecord]) -> dict:
        """
        Compute min/max summaries for numeric context fields.

        Non-numeric values are silently skipped.
        """
        all_keys: set[str] = set()
        for r in records:
            all_keys.update(r.context.keys())

        summary: dict[str, Any] = {}
        for key in sorted(all_keys):
            values = []
            for r in records:
                v = r.context.get(key)
                if isinstance(v, (int, float)):
                    values.append(v)
            if values:
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_max'] = max(values)
        return summary

    # ------------------------------------------------------------------
    # Flush strategies
    # ------------------------------------------------------------------

    def _flush_quiet(self, total_pairs: int | None) -> None:
        """Emit only critical-category warnings (aggregated)."""
        grouped = self._aggregate_by_category()
        for cat, records in grouped.items():
            if cat in _CRITICAL_CATEGORIES:
                msg = self._format_summary(cat, records, total_pairs)
                warnings.warn(msg, cat, stacklevel=2)

    def _flush_default(self, total_pairs: int | None) -> None:
        """Emit one aggregated summary per category."""
        grouped = self._aggregate_by_category()
        for cat, records in grouped.items():
            msg = self._format_summary(cat, records, total_pairs)
            warnings.warn(msg, cat, stacklevel=2)

    def _flush_verbose(self) -> None:
        """Emit every individual warning record."""
        for rec in self._records:
            warnings.warn(rec.message, rec.category, stacklevel=2)
