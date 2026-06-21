"""Hashable facts and the indexed ScriLog fact store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Tuple

@dataclass(frozen=True)
class ScriLogFactKey:
    """
    Immutable identity for a fact.

    Example:
        family(looped)

    becomes:

        predicate = "family"
        args = ("looped",)

    We separate FactKey from Fact because the key must be hashable
    and stable inside dictionaries/sets.
    """

    predicate: str
    args: Tuple[str, ...] = field(default_factory=tuple)

    def to_string(self) -> str:
        if not self.args:
            return self.predicate

        return f"{self.predicate}({', '.join(self.args)})"


@dataclass
class ScriLogFact:
    """
    A symbolic fact.

    Examples:
        loop_count(1)
        endpoint_count(2)
        family(looped)
        warning(possible_fragmentation)

    weight:
        Optional confidence/strength value.
        For v0.1 this is mostly informational.

    origin:
        Where the fact came from.
        Examples:
            "signature"
            "rule:family_looped"
            "profile:armenian_family_map"
    """

    predicate: str
    args: Tuple[str, ...] = field(default_factory=tuple)
    weight: float = 1.0
    origin: str = "unknown"

    @property
    def key(self) -> ScriLogFactKey:
        return ScriLogFactKey(
            predicate=self.predicate,
            args=self.args,
        )

    def to_string(self) -> str:
        return self.key.to_string()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "args": list(self.args),
            "weight": self.weight,
            "origin": self.origin,
            "text": self.to_string(),
        }


class ScriLogFactBase:
    """
    Tiny Prolog/Datalog-style fact store.

    This is the heart of ScriLog.

    It lets us say:

        facts.add("family", "looped")
        facts.has("family", "looped")
        facts.find("warning")

    But it stays fast because:
        - facts are finite
        - no full Prolog unification
        - no recursive search tree
        - every fact is indexed by a stable key
    """

    def __init__(self) -> None:
        self._facts: Dict[ScriLogFactKey, ScriLogFact] = {}

    def add(
        self,
        predicate: str,
        *args: Any,
        weight: float = 1.0,
        origin: str = "unknown",
    ) -> bool:
        """
        Add a fact.

        Returns:
            True  -> new fact was added
            False -> fact already existed

        If the fact already exists, we preserve it but update weight/origin
        when useful.
        """

        clean_predicate = str(predicate).strip()

        clean_args = tuple(
            str(arg).strip()
            for arg in args
            if arg is not None
        )

        if not clean_predicate:
            return False

        fact = ScriLogFact(
            predicate=clean_predicate,
            args=clean_args,
            weight=float(weight),
            origin=str(origin),
        )

        key = fact.key

        if key not in self._facts:
            self._facts[key] = fact
            return True

        existing = self._facts[key]

        # Keep the strongest known weight.
        if fact.weight > existing.weight:
            existing.weight = fact.weight

        # Preserve origin trail without making it too complex.
        if fact.origin not in existing.origin:
            existing.origin = f"{existing.origin}|{fact.origin}"

        return False

    def has(self, predicate: str, *args: Any) -> bool:
        """
        Check whether a fact exists.
        """

        key = ScriLogFactKey(
            predicate=str(predicate),
            args=tuple(str(arg) for arg in args),
        )

        return key in self._facts

    def find(self, predicate: str) -> List[ScriLogFact]:
        """
        Return all facts with a given predicate.

        Example:
            find("family")
            -> family(looped), family(two_endpoint), ...
        """

        wanted = str(predicate)

        return [
            fact
            for fact in self._facts.values()
            if fact.predicate == wanted
        ]

    def find_arg0(self, predicate: str) -> List[str]:
        """
        Return the first argument from all matching facts.

        Example:
            facts:
                family(looped)
                family(branched)

            find_arg0("family")
            -> ["looped", "branched"]
        """

        values: List[str] = []

        for fact in self.find(predicate):
            if fact.args:
                values.append(fact.args[0])

        return values

    def count(self, predicate: str) -> int:
        """
        Count facts with a given predicate.
        """

        return len(self.find(predicate))

    def as_strings(self) -> List[str]:
        """
        Export facts as sorted readable strings.
        """

        return sorted(
            fact.to_string()
            for fact in self._facts.values()
        )

    def as_dicts(self) -> List[Dict[str, Any]]:
        """
        Export facts as JSON-safe dictionaries.
        """

        sorted_facts = sorted(
            self._facts.values(),
            key=lambda fact: fact.to_string(),
        )

        return [
            fact.to_dict()
            for fact in sorted_facts
        ]

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self):
        return iter(self._facts.values())
