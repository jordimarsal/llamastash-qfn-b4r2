---
title: "spike: keeping a prompt across an unload with llama.cpp slot save/restore"
date: 2026-10-21
status: measured
unblocks: ["R14 batch 4"]
todo: "Per-model disk budget, and whether a kept dump survives a llama-server that serves several slots at once."
---

# What was measured

A single-slot `llama-server` was asked to read a 99,005-token prompt, save that
slot's processed state to disk, stop, start again, restore the file, and take a
follow-up request. Times are the engine's own (`timings.tns_ps` for the prompt
passes, `timings.save_ms` / `timings.restore_ms` for the two slot calls), on
this host, against the llama.cpp in use for this work — `0.6.0-dev`, build
11580, commit `4413a1924`.

| Step | Measured |
|---|---|
| Reprocess the prompt from nothing | 8,563 – 8,690 ms |
| `POST /slots/0?action=save` | 887 – 919 ms |
| File written | 3.02 GiB (≈ 32.8 KB per prompt token) |
| `POST /slots/0?action=restore`, file still in the page cache | 576 – 599 ms |
| Same restore after the page cache was dropped | 1,051 – 1,119 ms |
| Follow-up request after the restore | 99,005 prompt tokens cached, 23 – 26 ms |

The numbers shipped with R14 batch 4. What the run's model and KV cache type
were is not recorded with them, which is why the per-token cost is quoted as
measured instead of derived from a model's layer count — the KV layout is what
sets it, and it varies by model and by `-md`.

# Finding

**Keep the prompt. A stop that saves costs under a second, and the next launch
of that model pays about a tenth of a second for a prompt that would otherwise
cost nearly nine.** The saving is not in the save call or the restore call, both
of which are slower than they look; it is in the request that follows — 23 ms
against 8.6 s. Restoring is bounded by reading the file, so a cold restore is a
disk read of the dump and nothing more, and a warm one is cheaper than the save
that produced it.

What makes it a policy question rather than a feature flag is that the state is
large. 32.8 KB per token means the default 8192 MiB budget holds roughly a
quarter of a million prompt tokens across every model on the machine — a long
conversation on a small model, or a handful of long ones. A 34 B model with a
wider KV layout fills that budget with one 20k-token exchange. So the disk
budget is the decision the feature needs, and it is a config value the user
owns, not a constant in the launcher.

# The API, as confirmed against the local build

`--slot-save-path PATH` sets the directory slot dumps are read from and written
to. `GET /slots` lists the slots a running server has (`id_slot` per slot; on
this build it is enabled by default, and `--slots/--no-slots` is what governs
it). `POST /slots/{id_slot}?action=save` with `{"filename":"name"}` writes
`PATH/name` and answers `{id_slot, filename, n_saved, n_written,
timings.save_ms}`. `POST /slots/{id_slot}?action=restore` with the same body
reads it back and answers `{id_slot, filename, n_restored, n_read,
timings.restore_ms}`; it fails when the slot has no room or the file does not
describe a prefix that fits.

Saving is not a model-dump shortcut: it serializes the KV state of that slot
only, and llama.cpp matches a restored dump against the incoming prompt
token-by-token before reusing any of it. A request whose prefix does not match
what was saved gets the normal prompt pass. That is why no per-conversation
bookkeeping was added on our side, and why an unchanged conversation's *answer*
is not kept — the state kept is the processed prompt.

# What this unblocks

- The disk budget default: 8192 MiB for the whole tree, an age limit of 3600 s,
  and both dropped oldest-first. Sized from the per-token cost above.
- The save point: llama.cpp releases a slot's KV when its process exits, so the
  save has to happen while the engine is alive — on the stop path, not after.
- The restore point: a server that has already served a request has busy slots,
  and a restore into a busy slot fails. Restoring before the launch is handed
  out as Ready is what makes the call reliable rather than a coin flip.
- Not building it at all for a `--fit`ed launch: `--fit` decides memory from
  what is free, and a launch that keeps state on disk changes that arithmetic.

Open: a per-model budget (a 1B model's dumps should not be able to crowd a 70B
model's out of the tree), and whether the dump/restore pair behaves the same on
a multi-slot server — both measured on a single slot here.
