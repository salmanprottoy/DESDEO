# Domain glossary

## Decision experiment

One decision maker's interactive exploration of one multiobjective problem. It contains the DM's
preference inputs, the resulting state lineage, every solver-generated candidate considered in that
scope, and any saved or final selections.

## Experiment step

One recorded state-producing event in a decision experiment, such as initialization, candidate solving,
intermediate-solution generation, saving, or finalization. Steps may branch; their parent links define
lineage.

## DM-generated solution

A solution produced by the optimization method in response to a DM action or preference. The term does
not mean that the DM or an LLM numerically calculated the solution.

## Selected solution

The DM-generated solution currently in focus, identified by its experiment step and its index within
that step.

## Experiment context

The authorized, read-only facts of a decision experiment available for explanation: problem inputs,
DM preference history, experiment steps and lineage, stored solver results, saved/final selections, and
deterministic explanation artifacts.

## Evidence fact

An immutable value or deterministic derivation with a stable identifier and source provenance. Evidence
facts, not conversation text, support chat claims.

## Grounded chat answer

A read-only explanation or comparison rendered from verified evidence facts. It cannot create solutions,
change preferences, invoke optimization, save candidates, finalize a decision, or decide on behalf of the
DM.
