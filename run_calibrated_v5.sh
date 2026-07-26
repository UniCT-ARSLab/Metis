#!/usr/bin/env bash
# Continuazione PULITA di v3 dopo aver scoperto che il "success 0/20" era un BUG di misura del
# frozen eval (contava solo l'ultimo step; fixato in run.py con latch cumulativo). Re-eval di
# ckpt-2100 con codice corretto: 17/20 = 85% success (reward_mean 165.2 identico). La policy
# gia funziona: raggiunge posizione+orientamento e tiene la posa all'85% al gate corrente
# (4cm/12deg, hold 20 frame @ 0.30 -- vedi xarm_scenario.gd:142 che sovrascrive success_distance
# a 0.04 dopo ep1500).
#
# Reward INVARIATO rispetto a v3 (il cambio precision_distance e stato revertato: era basato sulla
# diagnosi sbagliata del gate a 2cm). Quindi riprendo da ckpt-2100 (main dir) CON replay-2100 =
# vera continuazione, stesso reward, nessun cold warmup.
#
# --best-metric success_rate: ora che il conteggio e corretto la success e reale (~85%), quindi e
# una metrica di best/recovery sensata E da all'auto-recovery un baseline valido (>5%). Obiettivo:
# consolidare/alzare la success al gate corrente con misura ora corretta.
cd /home/fedyfausto/Lavoro/GodotAgents/godot-gymnasium-keras-marl
exec python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --num-envs 8 --base-port 7200 \
  --num-episodes 8000 --max-steps-per-episode 300 --physics-frames-per-step 3 \
  --batch-size 256 --replay-warmup 20000 --random-exploration-episodes 40 \
  --resume --resume-checkpoint checkpoints/xarm_pose_calibrated_v3/ckpt-2100 \
  --critic-learning-rate 1e-4 --policy-update-every 4 \
  --critic-warmup-updates 1000 \
  --min-alpha 0.02 \
  --no-caps --grad-clip-adaptive --grad-clip-norm 10 \
  --auto-recovery --best-checkpoint --best-metric success_rate \
  --best-evaluation-every 100 --best-evaluation-episodes 20 --best-evaluation-training-episode 2200 \
  --collector-mode async --checkpoint-every 50 \
  --checkpoint-dir checkpoints/xarm_pose_calibrated_v5 \
  --headless --log-format compact --dashboard --dashboard-port 8770
