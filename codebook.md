# Behavioral Trajectory Parser Codebook

**Codebook identifier:** `behavioral-trajectory-codebook`  
**Parser identifier:** `behavioral-trajectory-parser`

This codebook defines the public parser output. A tool call is one event even when its shell payload contains multiple operations. Missing, ambiguous, unresolved, and unknown states are retained rather than coerced.

## Root and trajectory fields

- `parser_id`: immutable parser identifier.
- `codebook_version`: identifier of this codebook.
- `rollout_id`: caller-provided trajectory identifier.
- `repository_root`: trusted absolute root used only for path resolution, or null.
- `repository_root_source`: `cli`, `trajectory`, `roster`, `argument`, or `missing`.
- `events`: ordered parsed tool-call events.
- `diagnostics`: trajectory-level parser coverage diagnostics.

## Event identity and execution

- `event_id`: `<rollout_id>:<message-index>:<tool-call-index>`.
- `step_index`, `tool_call_index`: source positions.
- `original_tool`, `command`: observed tool and projected command text.
- `pairing_status`: `paired`, `missing`, or `ambiguous` observation pairing. An observation may be consumed by at most one tool call. Parent-only fallback is permitted only when its action contains exactly one tool call.
- `execution_status`:
  - `executed`: the call is paired with one environment observation; this does not mean success;
  - `rejected_invalid_arguments_attempt`: arguments could not be decoded;
  - `rejected_multi_command_attempt`: the harness rejected/collapsed multiple extracted commands;
  - `observation_missing`: no paired observation;
  - `parser_unknown`: ambiguous pairing or unsupported projection.
- `attempted_action_eligible`: false only for terminal submit/finish events.
- `semantic_composition_eligible`: an executed event with a supported semantic primary operation.

## Operation labels

`primary_operation` has exactly ten possible values: `EXPLORE_REPOSITORY`, `SEARCH_REPOSITORY`, `READ_FILE`, `MUTATE_FILE`, `RUN_FORMAL_TEST`, `INSPECT_WORKSPACE_CHANGES`, `PACKAGE_OR_BUILD`, `OTHER_OPERATION`, `NO_REPOSITORY_OPERATION`, and `UNKNOWN`.

- `contained_operations_ordered`: all recognized operations in left-to-right order.
- `primary_operation`: first contained operation other than `OTHER_OPERATION`; if none, the first contained value.
- `matched_rule_ids`: inspectable parser-rule provenance.
- `contains_formal_test`, `contains_mutation`, `contains_diff_inspection`: event-level operation indicators.
- `formal_test_invocation_count`: number of contained `RUN_FORMAL_TEST` operations.
- `contains_multiple_formal_test_invocations`: whether that count exceeds one.

An unresolved shell redirect adds `unresolved_redirect_destination` to `matched_rule_ids` but does not add `MUTATE_FILE` or a mutation path.

## Paths, reads, and mutations

- `affected_paths`: resolved repository-relative read/target paths.
- `mutation_paths`: resolved repository-relative mutation destinations.
- `external_redirect`: a resolved redirect outside the repository.
- `target_role`: `TEST_FILE`, `NON_TEST_REPOSITORY_FILE`, `MIXED`, `UNRESOLVED`, or `NOT_APPLICABLE`.
- `target_role_rule_ids`: role provenance; mutation destinations take precedence in compound events.
- `contains_non_test_repository_path`: true when a resolved mutation destination is non-test; false for a resolved pure-test mutation or non-mutation; null for a mutation without resolved destinations.
- `mutation_kind`: `CREATE`, `EDIT`, `DELETE`, `MIXED`, or `NOT_APPLICABLE`.
- `read_mode`: `DIRECT_FILE_READ`, `VIEWPORT_CONTINUATION`, `UNKNOWN`, or `NOT_APPLICABLE`.
- `read_scope`: `FULL_FILE`, `BOUNDED_RANGE`, or `NOT_APPLICABLE`.

Target roles use only repository-agnostic rules:

1. a conventional filename (`test_*`, `*_test`, `*_spec`, `*.test`, `*.spec`, or `conftest`, with an extension) is test;
2. a complete `test`, `tests`, `__tests__`, `spec`, or `specs` directory segment is test;
3. every other resolved path is non-test.

No repository-specific path exception is used. `path_role` normalizes with `posixpath.normpath` and does not strip leading dots.

## Result and success evidence

- `exit_code`: projected integer status or null.
- `command_succeeded_explicit`: true for exit zero, false for nonzero/error, null without explicit evidence.
- `no_observed_failure`: true/false for executed calls, null otherwise. It is not proof of success.
- `write_success_evidence`: `exit_code`, `no_observed_failure`, or null.
- `formal_test_result`: `passed`, `failed`, `unknown`, or `NOT_APPLICABLE`.
- `formal_test_result_rule`: decisive evidence rule.
- `formal_test_evidence`: all detected evidence tiers.
- `formal_test_result_conflict`: precedence-relevant conflict among status, pass, failure, or crash evidence.

Test-result precedence is post-summary process failure, strong failure summary, contextual failure trace, explicit nonzero status, strong pass summary, explicit zero status, loose failure signal, then unknown.

## Signatures

- `canonical_command_signature`: operation sequence, whitespace-normalized lowercase command, and sorted resolved paths. It supports deterministic repetition metrics; it is not semantic equivalence.

## Rollout metrics

- `operational_action_count`: attempted nonterminal events.
- `semantic_denominator`: executed supported semantic events.
- `exploration_share`: primary-operation explore/search/read events divided by the semantic denominator.
- `read_share`: primary reads divided by the semantic denominator.
- `search_share`: primary search events divided by the semantic denominator.
- `file_mutation_share`: primary mutation events divided by the semantic denominator; mutation is syntactically inferred and does not prove a filesystem state change.
- `build_package_share`: primary package-or-build events divided by the semantic denominator.
- `other_operation_share`: primary `OTHER_OPERATION` events divided by the semantic denominator.
- `non_test_repository_write_without_observed_failure_count`: executed mutation calls with at least one resolved non-test destination and no observed failure; a `MIXED` event counts once.
- `non_test_repository_write_without_observed_failure_share`: preceding count divided by the semantic denominator.
- `non_test_repository_write_with_explicit_success_count`: counted writes with explicit zero status.
- `non_test_repository_write_with_no_observed_failure_only_count`: counted writes supported only by absence of observed failure.
- `formal_test_call_count`: executed tool calls containing at least one formal-test operation.
- `formal_test_invocation_count`: contained formal-test operations in those calls.
- `formal_test_call_share`: formal-test calls divided by the semantic denominator; bounded by one.
- `formal_test_invocations_per_semantic_action`: formal-test invocations divided by the semantic denominator; may exceed one.
- `repeated_unsuccessful_command_count`: strict-consecutive retries; an event counts when it and the immediately preceding attempted event have the same signature and are unsuccessful.
- `recurring_unsuccessful_command_count_by_signature`: signature-specific failure-chain sensitivity; intervening signatures do not reset another signature, while success resets its own signature.
- `unproductive_tail_length`: attempted actions after the last counted non-test write or passing formal-test event; if neither exists, all attempted actions.
- `post_edit_diff_inspection`: whether workspace-diff inspection occurs after the final counted non-test write.
- `unverified_termination`: `no_write`, `verified`, or `unverified` according to whether a counted non-test write exists and a passing formal test follows it.
- `unverified_termination_after_non_test_repository_write`: compatibility boolean; true only for categorical `unverified`, false for both `no_write` and `verified`.
- `action_entropy`: unnormalized Shannon entropy of semantic primary operations.
- `action_entropy_normalized`: entropy divided by log of observed category count when defined.
- `action_switch_count`, `action_switch_rate`: changes between adjacent semantic primary operations.

Rejected invalid-argument and rejected multi-command events are unsuccessful for both repetition metrics. Missing-observation and parser-unknown events remain uncertain and do not count as failure or success.

Exploration uses primary operations while formal tests use contained operations, so these measures are not mutually exclusive.

## Diagnostics

- `events`, `applicable_target_events`.
- `compound_event_rate`.
- `unresolved_target_rate_applicable`.
- `mixed_target_role_rate`.
- `multiple_formal_test_invocation_event_rate`.
- `formal_test_result_conflict_rate`.
- `explicit_write_success_evidence_rate`, `weak_write_success_evidence_rate`.
- `unknown_primary_rate`, `other_operation_rate`.
- `ambiguous_pairing_rate`, `missing_observation_rate`.
- `top_unmatched_commands`: ten most frequent first tokens among `OTHER_OPERATION` events.

Undefined rates are JSON null, never zero.
