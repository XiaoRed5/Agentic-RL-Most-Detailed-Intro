# Failure Diagnosis

## 1. Bidirectional group saturation

The local run attempted 24 groups. Only four contained both successful and failed samples; twenty were all-zero or all-one. With group-relative normalization, their advantages are all zero. This is the dominant reason the run produced only eight optimizer steps.

## 2. Process signal is absent from a one-step task

The PRM-Lite diagnostic re-scored the same actions using 15 deterministic rules. Active groups remained 4/24. The outcome is expected: the task contains no error recovery, dependent reads, multi-turn tool errors, or long reasoning. Reward shaping cannot recover signals that the environment never exposes.

## 3. Seen-task learning without unseen transfer

Train accuracy increased by 8.3 points and same-task prompt holdout by 16.7 points. Fully unseen accuracy stayed at chance. This pattern is consistent with learning the mapping between a small set of policies and tool names, not a transferable tool-use strategy.

## 4. Restricted generation hides important failure classes

A four-way action token cannot produce:

- a nonexistent function name;
- malformed JSON;
- a wrong argument value;
- a missing required parameter;
- a premature natural-language answer;
- a repeated failing tool call;
- a failure to recover after a tool error.

These are precisely the behaviors tested by BFCL and TAU-2.

## 5. Evaluation mismatch

The local diagnostic uses exact next-action correctness. The paper uses final environment state for TAU-2 and executable exact match/state checks for BFCL-V4. Local accuracy must not be compared numerically to the paper table.

## Priority fixes

1. Connect unrestricted Qwen tool-call generation to the official evaluator.
2. Run deterministic BFCL smoke cases before training changes.
3. Introduce stateful multi-turn rollouts and response-token masks.
4. Re-run Vanilla first to establish a collapse curve.
5. Add PRM-Lite, LATA, and Joint under the fixed 3-seed matrix.
