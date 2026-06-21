"""Bounded Prolog-like forward-chaining engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from .constants import DEFAULT_MAX_RULE_PASSES
from .facts import ScriLogFact, ScriLogFactBase
from .signature import ScriLogSignature

RuleCondition = Callable[[ScriLogFactBase, ScriLogSignature], bool]
RuleAction = Callable[[ScriLogFactBase, ScriLogSignature], List[ScriLogFact]]


@dataclass
class ScriLogRule:
    """
    One symbolic rule.

    A rule has:

        name
        condition
        action

    Conceptually:

        IF condition(facts, signature) is true
        THEN action(facts, signature) emits new facts

    Example:

        IF loop_count >= 1
        THEN family(looped)

    This is Prolog-inspired, but evaluated as bounded Python logic.
    """

    name: str
    condition: RuleCondition
    action: RuleAction
    description: str = ""

    def apply(
        self,
        facts: ScriLogFactBase,
        signature: ScriLogSignature,
    ) -> Tuple[bool, List[ScriLogFact]]:
        """
        Apply one rule once.

        Returns:
            changed:
                True if the rule added at least one new fact.

            emitted:
                The facts produced by the rule action.
                Some emitted facts may already exist in the factbase.
        """

        if not self.condition(facts, signature):
            return False, []

        emitted = self.action(facts, signature)

        changed = False

        for fact in emitted:
            was_new = facts.add(
                fact.predicate,
                *fact.args,
                weight=fact.weight,
                origin=f"rule:{self.name}",
            )

            if was_new:
                changed = True

        return changed, emitted


class ScriLogEngine:
    """
    Bounded forward-chaining rule engine.

    This is the small ScriLog brain.

    It repeatedly applies rules until either:
        - no new facts are added
        - max_passes is reached

    Why bounded?
        To avoid Prolog-style search explosion.

    For v0.1:
        max_passes = 3 is enough.

    Example:

        Pass 1:
            loop_count(1)
            -> family(looped)

        Pass 2:
            family(looped) + family(branched)
            -> family(complex_loop_branch_endpoint)

        Pass 3:
            no new facts
            -> stop
    """

    def __init__(
        self,
        rules: Optional[List[ScriLogRule]] = None,
        max_passes: int = DEFAULT_MAX_RULE_PASSES,
    ) -> None:
        self.rules = rules or []
        self.max_passes = max(1, int(max_passes))

    def run(
        self,
        facts: ScriLogFactBase,
        signature: ScriLogSignature,
    ) -> Tuple[int, int]:
        """
        Run the rule engine.

        Returns:
            actual_passes:
                Number of passes performed.

            total_rule_fires:
                Number of rules whose condition was true.
                This is not the same as number of new facts.
        """

        total_rule_fires = 0
        actual_passes = 0

        for pass_index in range(self.max_passes):
            actual_passes = pass_index + 1
            changed_this_pass = False

            for rule in self.rules:
                changed, emitted = rule.apply(
                    facts=facts,
                    signature=signature,
                )

                if emitted:
                    total_rule_fires += 1

                if changed:
                    changed_this_pass = True

            if not changed_this_pass:
                break

        return actual_passes, total_rule_fires


def _fact(
    predicate: str,
    *args: Any,
    weight: float = 1.0,
    origin: str = "rule",
) -> ScriLogFact:
    """
    Small helper for creating rule-emitted facts.

    Example:
        _fact("family", "looped")
        _fact("warning", "possible_fragmentation")
    """

    return ScriLogFact(
        predicate=str(predicate),
        args=tuple(str(arg) for arg in args),
        weight=float(weight),
        origin=str(origin),
    )
