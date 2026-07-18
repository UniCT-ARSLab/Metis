# Esportare una policy Metis

Metis mantiene separati lo stato di training e la policy destinata all'inferenza.

## Artefatti prodotti dal training

Ogni salvataggio di un checkpoint aggiorna anche questi file dentro
`--checkpoint-dir`:

```text
checkpoints/nome_run/
├── ckpt-100.*
├── policy.keras
└── policy.json
```

- `ckpt-*` contiene optimizer, target network, critic e contatori necessari al resume;
- `policy.keras` contiene architettura e pesi della sola rete usata per scegliere azioni;
- `policy.json` descrive observation, action space, algoritmo e decoder degli output.

Il file `.keras` e' il formato di riferimento. I vecchi `.weights.h5` restano salvati
per compatibilita', ma richiedono che il codice ricostruisca manualmente l'architettura.

## Eseguire la policy Keras

`run.py` cerca automaticamente il bundle nella directory del checkpoint:

```bash
python/.venv/bin/python python/run.py \
  --checkpoint-dir checkpoints/nome_run \
  --godot-bin /percorso/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/mio_scenario.tscn
```

Non serve `--algorithm`: viene letto da `policy.json`. Per indicare un bundle diverso:

```text
--load-from policy --policy-path exports/mia_policy/policy.keras
```

`--policy-path` accetta anche un modello HDF5 completo (`modello.h5`) e i vecchi file
di soli pesi (`modello.weights.h5`). Un file senza `policy.json` non contiene il
contratto Metis: `run.py` verifica comunque la dimensione dell'input, ma nei casi
ambigui, per esempio PPO discreto contro DQN, bisogna specificare `--algorithm`.

## Avviare un nuovo training da una policy

La stessa opzione inizializza la rete decisionale di qualsiasi trainer:

```bash
python/.venv/bin/python python/train.py \
  --algorithm dqn \
  --policy-path exports/mia_policy/policy.keras \
  --checkpoint-dir checkpoints/nuova_run \
  --godot-bin /percorso/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/mio_scenario.tscn
```

Questo e' un **warm start**: episodio, optimizer, replay buffer, critic, target network
ed esplorazione ripartono dai valori iniziali. Per continuare davvero un training si
deve usare `--resume` o `--resume-checkpoint`; per evitare stati incoerenti, Metis non
permette di combinarli con `--policy-path`.

## Esportare in TFLite

TFLite e' incluso in TensorFlow e non richiede dipendenze aggiuntive:

```bash
python/.venv/bin/python python/export.py \
  --policy checkpoints/nome_run \
  --format tflite
```

Il comando crea `policy.tflite` accanto a `policy.keras` e aggiorna il manifest.
Sono disponibili tre modalita':

```text
--tflite-quantization none
--tflite-quantization dynamic
--tflite-quantization float16
```

`none` preserva float32 ed e' il default piu' prevedibile. `dynamic` riduce soprattutto
i pesi; `float16` puo' essere utile su dispositivi che lo accelerano. La quantizzazione
intera int8 richiede un dataset rappresentativo e non viene applicata implicitamente.

## Esportare in ONNX

ONNX usa dipendenze opzionali separate dal runtime di training:

```bash
python/.venv/bin/python -m pip install -r python/requirements-export.txt

python/.venv/bin/python python/export.py \
  --policy checkpoints/nome_run \
  --format onnx
```

Per produrre entrambi i formati:

```bash
python/.venv/bin/python python/export.py \
  --policy checkpoints/nome_run \
  --format all
```

L'export ONNX usa per default opset 18. Si cambia con `--onnx-opset`, limitato agli
opset 14-18 supportati dal convertitore scelto.

## Contratto di inferenza

Il modello non sostituisce il contratto dell'agente. Un'applicazione esterna deve:

1. costruire le observation nello stesso ordine e con la stessa normalizzazione;
2. passare un batch float32 di forma `[N, observation.size]`;
3. interpretare gli output usando `inference.decoder` in `policy.json`;
4. applicare nomi, limiti e struttura descritti nella sezione `action`.

I decoder attuali sono:

- `argmax_q_values`: DQN, seleziona l'indice Q massimo;
- `tanh_then_scale`: actor DDPG/TD3 e varianti;
- `tanh_mean_then_scale`: media deterministica dell'actor SAC;
- `ppo_deterministic_heads`: argmax delle head discrete e media delle head continue.

Critic, replay buffer ed esplorazione non fanno parte della policy di produzione.
