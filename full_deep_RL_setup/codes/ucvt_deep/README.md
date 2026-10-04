# UC-VT: Uncertainty-Calibrated Virtual Training (Deep Meta-RL, GPU)

Full SAC + TAVT-style task encoder + ensemble task decoder + UC-VT uncertainty
weighting, trained on two OOD task families built on Gymnasium/native MuJoCo
(`HalfCheetah-v5`, `Walker2d-v5`).

## Why this isn't a patch on the official `JM-Kim-94/tavt` repo

We initially planned to patch UC-VT into the official TAVT code. After
inspecting it, we found it depends on:
- legacy `gym` (not Gymnasium) + the deprecated `mujoco-py` C-extension, and
- for `walker-mass-ood`/`hopper-mass-ood` specifically, a bundled
  `rand_param_envs/mujoco_py` that performs **license-key registration
  against MuJoCo's old license server**, which no longer exists.

Patching that stack was a real risk of burning our remaining time on
environment/install failures rather than the actual research question. We
instead reimplemented UC-VT faithfully on the modern stack:
- `cheetah_vel` task family = same idea as `cheetah-vel-ood` (reward shift:
  target forward velocity), built on `HalfCheetah-v5`.
- `walker_mass` task family = same idea as `walker-mass-ood` (transition
  shift: agent mass), built on `Walker2d-v5` with body-mass scaling.

The task-preserving (bisimulation) loss is grounded directly in the official
repo's `compute_bisim_loss` (we read the source and reproduced its formula:
regress inter-task latent distance against inter-task reward + next-state
decoder-prediction distance). We deliberately **do not** reproduce the
official code's WGAN discriminator/generator refinement step or its
auxiliary context-distribution VAE — those are orthogonal engineering
machinery, not the UC-VT contribution, and out of scope for the time we had.
State regularization is implemented as a Lipschitz-style smoothness penalty
on the decoder's next-state prediction w.r.t. small task-latent perturbations
(same practical goal — don't let virtual-task predictions be erratic — via a
simpler mechanism). Both simplifications are noted so you can describe them
honestly as scope decisions in your report's limitations section.

## Files

- `envs.py` — `cheetah_vel` / `walker_mass` OOD task families, `set_task(z)`.
- `networks.py` — `TaskEncoder` (context -> latent z, KL-regularized),
  `EnsembleTaskDecoder` (K=5 bootstrap decoders -> uncertainty), SAC
  `TanhGaussianActor`, `TwinQCritic`.
- `losses.py` — SAC critic/actor losses, task-preserving (bisimulation) loss,
  state-regularization loss, UC-VT soft confidence weighting `w(u)`.
- `buffers.py` — per-task replay buffers + context sampling.
- `algorithm.py` — `UCVTAgent`: ties everything together, one `update()` call
  per gradient step, mixed precision (`torch.amp`), checkpoint state dicts.
- `train.py` — meta-training loop: rollout collection with posterior task
  inference, gradient updates, periodic near/mid/far OOD evaluation with
  on-the-fly context adaptation, checkpointing every N episodes (rotating,
  keeps latest 3).
- `config.py` — all hyperparameters in one place.

## Running on Kaggle

1. Enable GPU (Settings -> Accelerator -> GPU).
2. Enable internet (needed to `pip install gymnasium mujoco` and for MuJoCo's
   headless renderer deps — physics-only here, no rendering, so this is fast).
3. Upload these files to `/kaggle/working/` (or a Kaggle Dataset you attach).

```bash
pip install -q gymnasium mujoco
cd /kaggle/working
python train.py --task_family cheetah_vel --total_iterations 2000
python train.py --task_family walker_mass --total_iterations 2000
```

Always sanity-check a new environment/config with the smoke test first —
it runs the *entire* pipeline (rollout, update, checkpoint, eval) end to end
in under a minute on CPU, and will catch shape/NaN bugs before you spend GPU
time:

```bash
python train.py --smoke_test --task_family cheetah_vel
python train.py --smoke_test --task_family walker_mass
```

Checkpoints land in `cfg.checkpoint_dir` (`/kaggle/working/checkpoints` by
default), rotating to keep only the 3 most recent. Scalar logs (per-update
losses, per-eval-bin returns) are CSVs in `cfg.log_dir`.

## What we already verified before handing this to you

Both task families were run end-to-end (rollout -> update -> eval ->
checkpoint) on CPU with `--smoke_test` and produced finite losses with no
NaNs across 15 iterations. Two real bugs were caught and fixed this way and
are documented in code comments:
1. A tanh-Gaussian log-prob numerical instability (`1 - action^2` going
   fractionally negative from float32 rounding at saturation -> `log(negative)`
   -> NaN) — fixed by clamping before the log, not after.
2. The task-preserving loss's random task-pairing could sample the identity
   permutation, making the paired distance exactly 0, and `d/dx sqrt(x)`
   diverges at `x=0` -> NaN gradient. Fixed with a cyclic shift (guarantees
   distinct pairing) plus an epsilon under the square root as a second guard.

Full multi-thousand-iteration GPU training was **not** run (no GPU in this
environment) — budget real Kaggle time for a first short run (e.g. 200-300
iterations) to confirm learning curves look sane before committing to the
full 2000+ iteration run for your final numbers.

## Known simplifications to state explicitly in your report

- Task-preserving loss omits the WGAN refinement step and context-VAE from
  the official code (see above).
- State regularization is a smoothness penalty, not the official method's
  overestimation-correction mechanism.
- Actor updates use real transitions only (not virtual), for training
  stability; virtual transitions affect the critic only, via UC-VT weighting.
- `n_train_tasks=20`, `latent_dim=64` and other sizes in `config.py` are
  reasonable defaults for a course-project compute budget, not swept/tuned.
