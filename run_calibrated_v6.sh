#!/usr/bin/env bash
# STADIO HOLD-60: statico consolidato (v5 media 82.5%, picco 95% a ckpt-2400). Ora si allunga la
# tenuta. tscn XarmScenario: hold_curriculum_stage1_until=2400 -> per training episode>=2400 il
# curriculum entra in stage2 = hold 60 frame fisici (1s @60Hz) con velocita max 0.20 (piu fermo).
# stage2_until=999999 -> resta a 60 frame (il salto a 120/0.12 sara uno stadio successivo).
#
# --best-evaluation-training-episode 2600 (>=2400) cosi il FROZEN eval misura il hold-60, non il
# 20 facile: la success ora riflette la tenuta piu lunga.
#
# Resume da v5/best/ckpt-2400 (best success 95%). best/ non ha replay -> cold warmup 20k: VOLUTO,
# perche il cambio di curriculum altera la normalizzazione del hold reward (20->60 frame); il replay
# fresco evita di mescolare reward vecchie/nuove. Reward function INVARIATA. Settaggi conservativi
# invariati (critic-lr 1e-4, policy-update 4, min-alpha 0.02). --best-metric success_rate.
cd /home/fedyfausto/Lavoro/GodotAgents/godot-gymnasium-keras-marl
exec python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --num-envs 8 --base-port 7200 \
  --num-episodes 8000 --max-steps-per-episode 300 --physics-frames-per-step 3 \
  --batch-size 256 --replay-warmup 20000 --random-exploration-episodes 40 \
  --resume --resume-checkpoint checkpoints/xarm_pose_calibrated_v5/best/ckpt-2400 \
  --critic-learning-rate 1e-4 --policy-update-every 4 \
  --critic-warmup-updates 1000 \
  --min-alpha 0.02 \
  --no-caps --grad-clip-adaptive --grad-clip-norm 10 \
  --auto-recovery --best-checkpoint --best-metric success_rate \
  --best-evaluation-every 100 --best-evaluation-episodes 20 --best-evaluation-training-episode 2600 \
  --collector-mode async --checkpoint-every 50 \
  --checkpoint-dir checkpoints/xarm_pose_calibrated_v6 \
  --headless --log-format compact --dashboard --dashboard-port 8770
