# Dispatching the labelling (for the Claude session that runs it)

Work dir: `/data/datasets/unitree_g1_inspire/obj_queue`. 52 batches `batches/b000.txt` .. `b051.txt`, 20 sheets each
(the last one shorter; 1023 clips), plus the relabel batches `fix000` .. `fix004` (82 clips of `fix_clips.json`, 9-frame
sheets, 10-29) that `check_results.py` also checks. Rules for the labellers: `README.md`. The GPU side (SAM3 tracking of each box) is already
running on this machine and picks up every `results/bNNN.jsonl` as soon as it appears; do not start anything else.

Loop until `python3 check_results.py` prints `0 to do`:

1. `python3 check_results.py` -> the batches under `TODO:`.
2. Start up to **3 subagents at a time** of type **`obj-labeler`** (Opus, low effort, Read/Write only), one batch each,
   in the background. The prompt is just: `Batch bNNN.` (the agent definition says the rest). Give each agent a
   different batch; never give the same batch to two agents.
3. When an agent finishes, start the next TODO batch. If its reply says an image did not show, or `check_results.py`
   lists the batch as bad, delete `results/bNNN.jsonl` and give the batch to a fresh agent once more.
4. If the auto-approval blocks an agent's Write, ask it to paste the JSON lines in its reply and write
   `results/bNNN.jsonl` yourself (exactly those lines).

Lessons from the last run (AgiBot, 10-25): an agent that reads more than ~20 images starts losing the old ones and
guesses; more than 3–4 agents at once gave more blank images. One batch per agent, then a new agent.

When all batches pass, reply with the `check_results.py` summary.
