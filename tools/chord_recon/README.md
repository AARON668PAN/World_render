# Native CHORD reconstruction (records)

These files set up and test CHORD's ego reconstruction (hands + object from a human video) on the RTX 5090 without
Docker. The environment itself lives in the CHORD working copy, `/data/repos/chord/local/` (venv `.venv-recon`,
CUDA toolkit `.cuda128`, `third_party/` clones, `build/`). The scripts here are copies kept as a record; most of them
still point at that location. The full write-up is `docs/chord_recon_native.md`.

Paths in the doc and the scripts map to this folder like this:

| In the doc / scripts | In this repo |
|---|---|
| `/data/repos/chord/local/recon_env.sh` | `tools/chord_recon/recon_env.sh` |
| `/data/repos/chord/local/link_mano.sh` | `tools/chord_recon/link_mano.sh` |
| `/data/repos/chord/local/recon_reqs.txt` | `tools/chord_recon/recon_reqs.txt` |
| `/data/repos/chord/local/recon_smoke_partial.py` | `tools/chord_recon/recon_smoke_partial.py` |
| `$L/recon_patches/...` (`L=/data/repos/chord/local`) | `tools/chord_recon/patches/...` |
| build / download helper scripts in `$L/logs/` | `tools/chord_recon/setup/*.sh` |

`patches/trtexec_py.sh` runs the `trtexec_py.py` next to it with the recon venv (`RECON_VENV`, default
`/data/repos/chord/local/.venv-recon`). `recon_env.sh` still sets `TRTEXEC_PATH` to the copy in
`/data/repos/chord/local/recon_patches/`, which is the same file.

Status (11-01): MANO and the FoundationPose weights are in place, but the full reconstruction has not been run end to
end yet. The renderer's only human clip so far (CHORD's `tissue_box` sample) uses the processed parquet that ships with
CHORD, not an output of this env.
