# Task 05 — Close the fine-tuning loop for real

**Owner:** one agent. **Depends on:** nothing, works on the demo room today.

This is the reason the project exists, and it is the piece that is built but not yet run at volume.

## Where it stands

A qwen3.7-flash run on the demo room failed in an instructive way. Its recorded trace:

```
1 navigate_to pot   ok        4 place pot → person 2   ok  (handed the pot over instead of pouring)
2 pick pot          ok        8 pick pot               FAIL object_held
3 navigate_to 2     ok       10 pick pot               FAIL object_held  (repeats to max_steps)
```

Tags: `max_steps`, `policy_error:bad_json`, `precondition:object_held`, `precondition:unknown_target`.
The scripted oracle succeeds on the same room in 11 steps with no tags. So there is a real gap to close,
and the oracle knows the right answer at every one of those states.

## Do this

1. **Run a batch.** Write `tools/run_batch.py`: N seeds × the `chat` policy, headless, no browser. The
   runtime already runs faster than real time when nothing renders. Log success rate and the tag
   histogram. Do not skip this — a tag histogram over 50 episodes tells you what to fix.
2. **Export.** `POST /sft/{job}` or `python tools/make_sft.py jobs/test/episodes --out sft.jsonl`.
   Every failed step is relabeled with what `policy.oracle` would have done from that exact state, which
   is DAgger with a free expert. Steps where the oracle would have made the same mistake are dropped as
   having no learning signal.
3. **Fine-tune.** Any provider that eats OpenAI chat JSONL. Keep `policy.SKILL_SYSTEM` byte-identical
   between training and serving; it is the system message in every example.
4. **Rerun and compare.** `POST /episode/start` with `replay_of: <episode_id>` copies task, seed, robot
   and crowd schedule and swaps only the policy. The UI groups reruns under the original. Report before
   and after success rates.
5. **Watch for the obvious trap.** The oracle only knows `coffee_to_person`. For `bring_object` and
   `go_to` the relabeler must run with `--relabel none`, which means only human demonstrations and
   successful runs are usable. Either write oracles for those task types (see Task 06) or say so in the UI
   when the button is pressed.

## Done when

Two numbers in this file: success rate over 50 episodes before fine-tuning and after, on the same seeds.
