#!/usr/bin/env bash
# STADIO HOLD-120 (finale del curriculum di tenuta): hold-60 consolidato (v6 ep3200 = 95%,
# reward 192, ckpt-3200). Ora tenuta 120 frame fisici (2s @60Hz) con velocita max 0.12 (il piu
# fermo). tscn XarmScenario: hold_curriculum_stage2_until=3200 -> per training episode>=3200 il
# curriculum entra in stage3 = 120 frame/0.12 (stage1_until=2400 invariato).
#
# --best-evaluation-training-episode 3300 (>=3200) cosi il FROZEN eval misura il hold-120.
#
# Resume da v6/best/ckpt-3200 (best 95%). Cold warmup 20k voluto (cambio curriculum altera la
# normalizzazione hold reward 60->120; replay fresco evita mix). Reward INVARIATA. Settaggi
# conservativi invariati. --best-metric success_rate.
cd /home/fedyfausto/Lavoro/GodotAgents/godot-gymnasium-keras-marl
exec python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --num-envs 8 --base-port 7200 \
  --num-episodes 8000 --max-steps-per-episode 300 --physics-frames-per-step 3 \
  --batch-size 256 --replay-warmup 20000 --random-exploration-episodes 40 \
  --resume --resume-checkpoint checkpoints/xarm_pose_calibrated_v6/best/ckpt-3200 \
  --critic-learning-rate 1e-4 --policy-update-every 4 \
  --critic-warmup-updates 1000 \
  --min-alpha 0.02 \
  --no-caps --grad-clip-adaptive --grad-clip-norm 10 \
  --auto-recovery --best-checkpoint --best-metric success_rate \
  --best-evaluation-every 100 --best-evaluation-episodes 20 --best-evaluation-training-episode 3300 \
  --collector-mode async --checkpoint-every 50 \
  --checkpoint-dir checkpoints/xarm_pose_calibrated_v7 \
  --headless --log-format compact --dashboard --dashboard-port 8770
