extends Node3D
## xArm scenario whose target is a RIGID BODY spawned anywhere inside a spawn area.
##
## Separate from xarm_scenario.gd on purpose. That script serves the reach scenes: its target is a
## Marker3D picked from a pool of spawn markers, and it binds `$Target` directly. Here the target is
## a RigidBody3D that has to be teleported through the physics server and set down on the floor, and
## the spawn positions are a continuous area rather than a list of points. Bolting both behaviours
## into one script would have put a second, mutually exclusive mode inside every reach run.
##
## Two things this scene adds over a plain reach:
##
## 1. The object spawns UNIFORMLY inside the spawn area, resting on the floor.
## 2. The demanded grasp orientation FOLLOWS THE AZIMUTH of the object as seen from the arm base.
##    Without that, the demanded orientation is world identity everywhere, so the wrist has to
##    absorb the whole azimuth sweep across the area by itself -- on the OpenArm that made a fixed
##    angle gate free at one end of the workspace and unreachable at the other, and cost thousands
##    of episodes before it was recognised.

@export_category("Grasp target")
## The rigid body to place. Its child marker is the pose the arm is asked to reach.
@export var grasp_target: RigidBody3D
@export var grasp_target_pose: Marker3D
## THE CUBE CANNOT TURN. Its three angular axes are locked in the scene, and that is deliberate.
##
## A knocked cube used to yaw and tip -- measured 29 degrees on a scripted grasp, 94 when only the
## vertical axis was locked and the energy went into tumbling instead. A 30 mm cube turned 45 degrees
## presents a 42 mm diagonal, so the claws that fit around it at rest stop fitting, and the gap test
## refuses grasps for a reason that has nothing to do with what the policy did. Orientation is out of
## this task by decision, so there is nothing a turned cube can teach and plenty it can hide.
##
## The demanded grasp pose was never the problem: `grasp_target_pose.global_basis` is re-pinned from
## `_demanded_basis` every physics frame, so the marker's orientation was already held steady while
## the body under it spun. The cube's own footprint was the thing that moved.
##
## Honest cost: on the scripted IK run this went from 2 captures out of 3 to 1, because a cube that
## cannot absorb a shove by rotating slides further instead (166 mm against 147). The shoving itself
## is the real defect and locking rotation does not touch it.
##
## Fastest the loose cube is allowed to travel, in m/s.
##
## A kinematic claw does not push, it teleports: the pad geometry is rewritten from forward
## kinematics every frame, so when it lands inside the cube the solver resolves the overlap with an
## impulse that has nothing to do with how fast the arm was really moving. Measured, that launched
## the cube 32 metres and then 86 metres straight down past the edge of the floor plate, in 9 of 88
## episodes. Capping the speed is not cosmetic: an episode whose cube is in orbit teaches nothing and
## still costs 300 steps.
@export var cube_max_speed := 1.0
## Puts the cube back on the surface if it ever ends up under it.
@export var cube_floor_rescue := true
## Physics frames after a reset during which the cube is pinned to the spot it was spawned on.
##
## The rung is solved against the cube WHERE IT WAS PLACED, and then the agent poses the arm --
## joint positions written directly, so the links appear at the seeded pose without sweeping there.
## On 38% of spawns that appearance overlapped the cube for a frame and the solver shoved it 20-30 mm
## clear before step 0 ever ran, which is why a rung solved to 2.9 mm was measured at 21.6 mm once the
## episode started. The push is a one-off: the cube then sits still. Pinning it across those frames
## makes the seed mean what it says.
@export_range(0, 20, 1) var cube_pin_frames := 6
## Whether the cube may tumble once it has been let go.
@export var cube_free_after_release := true

## Area whose collision box bounds the spawn positions. Only x and z are sampled from it: the
## resting height is derived from the object's own collision shape and the surface below it.
@export var spawn_area: Area3D
## Surface the object is set down on. Its top face gives the resting height.
@export var spawn_surface: Node3D
## Keeps the object clear of the area edge by its own half width, so it never spawns half off.
@export var spawn_edge_margin := 0.005

@export_category("Azimuth")
## Rotate the demanded orientation so its forward axis points along the object's direction from the
## arm base, instead of demanding world identity everywhere.
@export var target_yaw_follows_azimuth := true
@export_range(0.0, 1.0, 0.05) var target_azimuth_yaw_gain := 1.0
## Extra random yaw on top of the azimuth, so the policy learns a pose rather than one convention.
@export_range(0.0, 180.0, 1.0) var target_yaw_randomization_degrees := 0.0
@export var target_yaw_randomization_start_episode := 0

@export_category("Grasp capture")
## How close the tool has to be to the object before the claws can take it.
@export var grasp_capture_distance := 0.03
## Fraction of the gripper travel towards shut required to count as closed on the object.
@export_range(0.0, 1.0, 0.01) var grasp_capture_closure := 0.6
## Claw opening the tool point is frozen at, as a fraction of travel towards shut.
##
## 0 keeps it exactly where it used to sit at reset with the claws open, so every pose the seeding IK
## and the reward were tuned around still holds; the only thing that changes is that it stops sliding
## forward as the claws shut. Freezing it at the closure the cube is actually held in centres it
## better on the cube (2.9 mm against 19 mm) but moves the whole approach pose, and at 0.6 the seed
## drove xarm_4_link into the floor and every episode started terminal.
@export_range(0.0, 1.0, 0.01) var tool_frozen_closure := 0.0
## An object still moving has been knocked, not grasped.
@export var grasp_capture_max_object_speed := 0.05
## How far the claws must reopen past the capture threshold before a held object is let go.
@export_range(0.0, 1.0, 0.01) var grasp_release_clearance := 0.15
## Note on `gripper_stops_on_contact`, which is now ON in the agent scene, having been off for most
## of this scene's life.
##
## It genuinely could not work before: the shared agent decides "object in the pad path" from the
## object's offset in the TOOL frame, and the tool used to ride the claw tips and travel 29 mm as
## they shut, so the object left the window during the closure and the test read false at every step.
## Freezing the tool point in the hand frame removed that, and the flag was re-measured rather than
## left off on the strength of a stale note.
##
## With it off the closure SQUEEZED THE CUBE OUT: the pads sweep in an arc, they never stop, and from
## a seed that placed the cube dead centre (0.0 mm across) they shoved it to 13.7-21.3 mm by the end
## of the closure -- outside a window that shrinks to nothing as the gap reaches the cube's width.
## Measured over ten seeds, captures went from 4 to 7, and the scripted run from one delivery to two.
##
## Decide the grasp from the claw geometry rather than from proximity plus a closure fraction.
## False restores the old rule exactly, so this is one switch to undo.
@export var grasp_capture_geometric := true
## Slack added to the measured claw window when testing whether the object is inside it.
##
## 10 mm, not 5: the reverse-curriculum seed places the claws by IK to about 3 mm, plus the tool
## inset, and measured across three seeds one of them left the cube 5.1 mm off centre against a
## 5.0 mm limit -- a perfectly good grasp refused by a hair.
@export var grasp_capture_tolerance := 0.010
## How far past the object's width the gap may still be and count as closed onto it.
@export var grasp_capture_gap_tolerance := 0.004
## How far the pads may close PAST the object before the grasp stops counting, in metres.
##
## The gap test used to be one-sided -- rejected only when the pads were still too far apart -- on the
## stated grounds that "a kinematic claw cannot be stopped by contact, so it always ends up inside the
## object". That has been false since `gripper_stops_on_contact` was switched on. One-sided, a claw
## shut all the way through the cube passed trivially: gap 4.4 mm against a 30 mm cube reads as
## "closed onto it", and the cube ends up pinned at the KNUCKLE with the claws shut, which is exactly
## what it looked like on screen.
##
## Measured over eight seeded grasps, a real capture sits at gap - width = -0.3 mm, the pads right on
## the cube; the one bad case in that set read -21.4 mm. 6 mm keeps every good grasp and rejects the
## claws-passed-through ones.
@export var grasp_capture_gap_overclose := 0.006
## How far BEHIND the claw tips the cube may sit and still count as between them. The tips advance
## as they close, so a held cube ends up this side of them; measured, about 29 mm of that is the
## closing travel alone.
## How far BEHIND THE LIVE TIPS the object may sit and still count, in metres.
##
## Measured from the tips as they actually are, not from where they were when open. Measured on a
## seeded grasp, a good capture puts the cube 19-20 mm behind the closed tips -- the tips are the
## deepest point of the claw, and the pad faces that actually hold the cube are set back from them.
## 0.020 sat exactly on that value and cut the seeded captures from 8 to 2; 0.030 covers it with
## margin and still stops well short of the knuckle, where 60 mm used to let the cube sit.
@export var grasp_capture_depth := 0.03
## How far PAST the tips the object may be and still count, in metres.
##
## This used to borrow `grasp_capture_distance`, which is a sideways tolerance, and so allowed the
## cube to sit 30 mm BEYOND the fingertips -- outside the claws altogether, which is exactly what a
## bad grasp looks like on screen. A good capture measures -19 to -21 mm, comfortably behind the tips,
## so a few millimetres of margin in front is all that is wanted: the object belongs BETWEEN the
## claws, never ahead of them.
@export var grasp_capture_ahead := 0.005
## Extra gap, on top of the capture tolerance, before a held object is let go.
## How much wider than the capture gap the pads must open before the object is let go, in metres.
##
## 0.006 was thin enough that a few millimetres of jitter on the gripper command dropped the cube
## mid-carry. 0.015 gives the hold real hysteresis: the claws have to open deliberately, not merely
## wobble.
@export var grasp_release_gap_clearance := 0.015
## NOTE: the PrematureClosure reward node is weighted 0.0 in the scene. It was the last term paying
## the policy to keep the hand OPEN -- up to 0.1 per step over 300 steps, against a one-off +15 for
## actually capturing -- and with the reverse curriculum starting the episode ON the cube it punished
## almost exactly the behaviour being taught. Measured over 17000 episodes with it on: the gripper
## command stayed at +0.135 (net opening) and the gap never left its fully-open stop, 53.2 mm of a
## 54.0 mm maximum. Restore the weight if the arm ever starts arriving clenched.
##
## Distance at which the premature-closure penalty reaches full strength. Between the capture radius
## and this it fades, so there is a band where trying to close is cheap. With a hard step at the
## capture radius instead, the only place closing was free was the one place the policy had not
## learned to reach yet, and closing anywhere else cost as much as the goal was worth.
@export var grasp_premature_closure_full_distance := 0.12
## Horizontal displacement from the spawn that costs the full disturbance penalty.
@export var grasp_disturbance_scale := 0.005
## Squeeze depth that costs the full penetration penalty.
@export var grasp_penetration_scale := 0.01 # non usato su questa mano: niente geometria dei pad
## Depth below the surface at which the claws pay the full digging penalty.
##
## The claw links are exempt from the floor collision on purpose -- a cube resting on the ground
## cannot be grasped by claws forbidden to reach it -- and that exemption left NOTHING charging them
## for going under. Measured on 20 episodes of v18: the lowest tip sat at +4.4 mm typically but
## reached -27.3 mm, below the floor in 4 of 20. The cube's centre is 15 mm up, so nothing useful
## happens down there; it is wasted behaviour that also looks wrong.
##
## Graded and NON-terminal by design. Terminating would put back exactly the rule that made the
## exemption necessary: the claws have to be allowed to reach y=0.
## 0.015, non 0.02: la penalita' satura prima, cosi' scendere di 17 mm costa il massimo invece di
## 25 mm. Misurato sulle 242 registrazioni dell'esperto, la punta piu' bassa ha mediana +0.2 mm e
## solo il 4% degli episodi scende sotto -12 mm, quindi la saturazione a 17 mm resta fuori da quello
## che una presa buona fa davvero.
@export var claw_floor_penalty_scale := 0.02
## Millimetres below the surface the claws may use for free.
##
## Not zero: the cube's underside IS the floor, so closing around it puts the tips at y=0 and a
## couple of millimetres under. Charging that taxes the grasp itself -- with no margin the seeded
## capture went from 14.47 to 12.29 against 7.01 for standing still.
@export var claw_floor_free_margin := 0.005
## Quanto la posa a fine episodio puo' discostarsi da quella iniziale prima che il costo saturi.
##
## Chiesto: il ritorno a casa deve finire nella STESSA posa di partenza, non solo con l'utensile nello
## stesso punto. Sono cose diverse -- lo stesso punto si raggiunge con il gomito ribaltato o la base
## girata altrove -- e finora era misurata solo la posizione dell'utensile.
## 2.0 rad, non 0.6. La scala e' il punto in cui il termine SATURA, e con 0,6 (34 gradi) l'errore
## reale della base -- misurato +103,1 gradi, cioe' 1,8 rad -- stava tre volte oltre: il termine
## valeva zero comunque, quindi migliorare da 103 a 80 gradi non pagava NIENTE. Bloccare il successo
## sulla posa non e' bastato, perche' togliere un premio che non si prendeva comunque non crea un
## gradiente. Con 2,0 rad il termine e' graduato su tutto l'intervallo in cui l'errore vive davvero.
@export var home_pose_penalty_scale := 2.0
## Velocita' dei giunti oltre la quale il costo del "non fermarsi a casa" satura.
##
## Misurato su v53: dopo la consegna il braccio arriva a 31 mm dal bersaglio -- il gate ne chiede 30 --
## ma ci arriva a velocita' 0.83 quando il gate di immobilita' ne chiede 0.15. Non e' che non ci
## arrivi: e' che non si ferma. Niente lo pagava per decelerare.
@export var home_still_scale := 0.8
## Fa cominciare l'episodio A CONSEGNA GIA' FATTA: cubo nella zona, braccio sopra di essa, bersaglio
## casa. Serve ad allenare il SOLO ritorno.
##
## Perche' isolarlo: presa, trasporto e consegna funzionano all'85% partendo da casa, ma il ritorno
## e' rimasto a zero attraverso quattro forme diverse di reward. Cercarlo in fondo a una catena di
## 180 passi vuol dire che la policy lo incontra solo dopo aver fatto tutto il resto bene, cioe'
## poche volte e tardi. Da solo e' un compito corto e denso, con l'unica cosa da imparare in primo
## piano.
@export var start_after_delivery := false
## Quanto il giunto peggiore puo' discostarsi da casa perche' il ritorno conti come finito.
##
## Senza questo "a casa" vuol dire solo che il PUNTO dell'utensile e' a posto, e siccome quel punto
## sta sull'asse di rotazione la base puo' essere ovunque: misurato, +103 gradi in ogni episodio.
@export var home_pose_tolerance_degrees := 15.0
## Con false il compito della POLICY finisce alla consegna, e il rientro non e' piu' suo.
##
## Il ritorno a casa e' l'unica parte di questo compito che non ha bisogno di essere imparata: sul
## robot vero si comandano i giunti di casa e ci vanno. Misurato su sette forme diverse di reward, la
## policy migliore chiude il giunto peggiore a 21-52 gradi (mediana 38,9, successo 0 su 9), mentre le
## stesse azioni dell'esperto rigiocate in anello aperto arrivano a 12-23. L'esperto ci riesce perche'
## e' un IK ad anello chiuso che itera fino al successo, non perche' sappia qualcosa in piu'.
##
## Il rientro resta nel compito -- il registratore lo dimostra e sul banco si esegue -- ma come
## comando di posizione, non come qualcosa da scoprire per tentativi.
@export var home_return_required := true
## Errore di posa, in radianti, oltre il quale il ritorno vale zero progresso.
##
## 2.0 rad copre l'arco che la base percorre davvero: da sopra la zona sta a 103 gradi (1,8 rad) da
## casa, quindi una banda piu' stretta lascerebbe piatta la prima meta' del viaggio -- lo stesso
## difetto che la banda da 150 mm aveva sulla distanza.
@export var home_return_pose_span := 2.0
## La posa in cui il rilascio lascia davvero il braccio, in gradi rispetto a casa, base per prima.
##
## MISURATA sull'esperto al momento del rilascio, su quattro episodi: +88.5/+89.2, -23..-27, +67..+71,
## +54, ~0, -22.9. La semina del solo ritorno la risolveva invece con l'IK su un punto sopra la zona,
## e ne usciva +88.5 -20.3 +54.4 +54.7 +4.4 +0.0 -- diciassette gradi di scarto sul terzo giunto e
## ventitre sulla pinza. Allenare il ritorno da una posa che il compito intero non produce insegna un
## tratto che poi non si incastra: e' il punto sollevato guardando le registrazioni.
@export var release_pose_degrees := PackedFloat32Array([88.9, -25.3, 69.2, 54.3, -0.2, -22.9])
## Quanto la posa di partenza del ritorno viene variata, in gradi, perche' il tratto non sia allenato
## su un solo punto.
## 15 gradi, non 4. Il clone del ritorno ha val_mse 0,000937 -- imitazione quasi perfetta -- e in
## anello chiuso fa 0 su 20, fermandosi a 252 mm dal bersaglio. Con 54 dimostrazioni da 56 passi e
## +-4 gradi di variazione, il tubo di stati coperto e' strettissimo: appena la policy ne esce di un
## soffio non ha piu' niente da imitare, ed e' l'errore che si accumula. Allargare la partenza allarga
## il tubo, che e' il rimedio diretto a quel difetto.
@export var release_pose_jitter_degrees := 15.0
## Con false il cubo, una volta catturato, NON viene incollato all'utensile: resta un corpo dinamico
## e a tenerlo devono essere l'attrito e la stretta delle chele.
##
## Vero e' come ha sempre funzionato questo compito: la cattura dichiara la presa e il cubo viene
## congelato nel frame della mano. Tutte le misure fatte finora poggiano su quello, quindi il
## passaggio si prova PRIMA sul registratore e si confronta la resa, invece di cambiare il compito e
## scoprire dopo che cosa e' cambiato.
@export var grasp_attach_kinematic := true
## Below the surface, in metres, at which the claws stop being merely charged and become a TERMINAL
## breach -- when there is NOTHING to grasp within reach.
##
## The graded penalty above cannot make sinking impossible: it saturates at claw_floor_free_margin +
## claw_floor_penalty_scale = 25 mm under, so a further metre costs exactly the same as 25 mm.
## Nothing pulls the arm back out, which is why the claws still went visibly through the ground.
##
## DEPTH ALONE CANNOT DECIDE THIS, and that was the first thing tried. Measured with
## tools/local/xarm_grasp_zone_floor_breach_probe.gd: a grasp that SUCCEEDS puts the tips as deep as
## -16.3 mm (claw geometry -26.3 mm), while driving every joint into the ground reaches only
## -23.7 mm. The two overlap, so every threshold that forbids the digging also kills real captures --
## a 15 mm rule cost seed 64, which captures cleanly without it.
##
## What separates them is not how deep but WHY: the exemption exists so the claws can close around a
## cube resting on the floor, and in that act the cube sits 20.5-31.6 mm from the tip midpoint. While
## digging it is 329.8 mm away -- a factor of ten, with no overlap at all. So depth is bought with
## proximity to the cube, and this shallow limit applies whenever there is nothing there to grasp.
@export var floor_breach_depth := 0.005
## The deeper allowance, granted only while the cube is within floor_breach_grasp_radius.
##
## Nearly twice the deepest successful grasp on record (-16.3 mm), so the rule has room before it
## touches anything real.
@export var floor_breach_grasp_depth := 0.030
## How close the cube must be to the tip midpoint for the deep allowance to apply, in metres.
##
## 100 mm: three times the worst measured grasp reach (31.6 mm) and a third of the digging (330 mm).
@export var floor_breach_grasp_radius := 0.10
## Below the surface, in metres, at which a HELD cube counts as pushed through the floor.
##
## Tight, because unlike the claws it has no legitimate excursion: the cube's underside IS the floor
## when it rests, and a grasp squashes it about a millimetre down.
@export var floor_breach_object_depth := 0.010
## Set false to keep measuring the breach without ending the episode.
@export var floor_breach_terminates := true
## Il blocco che regge la zona di rilascio, trattato come il pavimento una volta che il cubo e' su.
##
## Visto a schermo: il braccio tocca la sporgenza mentre trasporta il cubo e l'episodio continua. Due
## cause, misurate: i link delle chele sono esenti dalle collisioni con l'ambiente -- serve per
## arrivare sotto al cubo, ma l'esenzione non si spegneva mai -- e il cubo in mano non e' un link del
## robot, quindi veniva controllato solo contro il piano del pavimento e mai contro il blocco, la cui
## cima sta a 50 mm. Il cubo consegnato appoggia a 71.5 mm, quindi 21.5 mm di margine: la regola
## vieta di attraversare la sporgenza senza vietare la consegna.
@export var ledge_breach_terminates := true
## Quanto il cubo portato puo' scendere sotto la cima del blocco prima che conti come sfondamento.
## 25 mm, non 5. La regola serve a impedire che il braccio ATTRAVERSI il supporto, non a sorvegliare
## un avvicinamento basso. A 5 mm il corridoio della consegna era largo 8 mm -- il cubo consegnato
## appoggia a 77.3 mm di base contro un limite a 69.2 -- e una policy che prova a posare piano invece
## di lasciar cadere moriva li'. Misurato in v51: 112 prese su 200 episodi e ZERO consegne, dopo di
## che anche le prese sono svanite, perche' prendere non portava piu' da nessuna parte.
@export var ledge_breach_object_depth := 0.025
## Tilt past which the object counts as knocked over.
@export var grasp_tip_degrees := 60.0

## IL PAVIMENTO NON E' UN OSTACOLO, ed e' una scelta forzata, non una dimenticanza.
##
## Il rilevamento nell'agente registra solo i corpi nel gruppo `robot_obstacle`, e il pavimento non
## ci sta: misurato, il braccio scende a y=-0.102 (dieci centimetri SOTTO il piano) senza che venga
## segnalata una sola collisione, in ogni run. La penalita' di -20 e la terminazione non scattano.
##
## Mettendolo nel gruppo il rilevamento funziona -- 1 collisione, episodio chiuso al passo 32, il
## braccio si ferma a y=0.021 invece di sprofondare -- ma la presa diventa IMPOSSIBILE: la stessa run
## IK che prendeva il cubo 3 volte su 3 si ferma a chiusura 0.22-0.25 senza mai catturare, perche' le
## chele devono scendere attorno a un cubo il cui centro sta a 15.5 mm dal suolo.
##
## Le due cose non stanno insieme finche' l'oggetto e' appoggiato a terra. La via d'uscita e'
## rialzare il cubo su un piano, non spegnere il controllo. Vedi tools/local/xarm_grasp_zone_floor_collision_probe.gd.
##
@export_category("Tool frame")
## Put the tool frame on the claws' PINCH POINT, measured from their collision geometry, instead of
## at a hand-written offset from the wrist.
##
## `tcp_local_offset` was (0, 0.1, 0), which sits 73 mm past the claws: an object really held was
## far outside the 30 mm capture radius, and an object inside the radius was nowhere near the claws.
##
## Tracked live, because the claws swing: the tip midpoint runs y +0.1153 (shut) to +0.0863 (open)
## in the hand frame, 28.9 mm of travel that stays between the claws throughout.
##
## Worth knowing which links this reads. `gripper_finger_link_names` used to name grip_left/right --
## the linkage arms, which stop 50 mm short of the claws. Everything derived from them was wrong by
## that much, and the measured pad gap came out at 3.7 mm, narrower than the 30 mm cube. Read off
## finger_left/right, the real claws, the gap is 54.0 mm and the cube fits.
##
## Done here rather than in urdf_robot_arm_agent.gd because that script is shared with the OpenArm,
## whose gripper does not have this problem.
@export var tool_pose_follows_finger_tips := true
## How far back from the very tips the grasp point sits, along the hand axis.
##
## The tip midpoint is the extreme end of the claws. An object is held a little inside that, between
## the inner faces, so the point the arm aims at is pulled back by this much. Tuned by eye against
## the rendered claws; the sphere in tools/local/xarm_gripper_marker_demo.gd shows exactly where it lands.
## Shift of the frozen tool point along the hand axis, in metres. Negative moves it FORWARD, towards
## where the pads close.
##
## This is the point the whole reward is measured from -- the grey sphere on screen -- and it does not
## move when the claws shut, deliberately. But it was sitting ~8 mm short of where a properly held
## cube ends up: measured at capture, the cube reads +7 to +9 mm in FRONT of it. So "put the tool on
## the cube", which is what the reward asks for, left the pads about a centimetre high, and the arm
## hovered OVER the cube instead of wrapping it. Moving the point forward by that measured 8 mm makes
## the reward's target and the grasp's working point the same place.
@export var tool_tip_inset := -0.003

@export_category("Drop")
## Second half of the task: once the cube is held, carry it to the drop zone and let it go there.
##
## False leaves the scene exactly as it was -- pick the cube up and hold it -- so the two halves can
## be trained and debugged apart. That separation is deliberate: training both before either works
## is how the OpenArm spent months on failures nobody could attribute to one half or the other.
@export var drop_enabled := true
## The region the cube has to end up in. Its collision shape gives the footprint.
@export var drop_zone: Area3D
## How high above the zone the arm is asked to bring the cube before opening the claws.
@export var drop_release_height := 0.045
## Height the cube's centre must reach while held before a delivery counts, in metres.
##
## Watching a render, the arm was DRAGGING the cube along the floor to the zone. Nothing asked for a
## lift: the carrying progress only measures distance to the drop pose, and a low path is barely
## longer than a high one, while the delivery test had no vertical bound at all. So sliding it there
## scored exactly like carrying it there.
##
## Measured against the scene as it now stands, not chosen: the drop zone was raised onto a support
## whose top is at y=0.050 and which is itself in `robot_obstacle`, and the delivery surface is at
## y=0.0715, so a cube resting there sits at 0.0865. A 45 mm requirement -- below the obstacle it is
## supposed to clear -- would have been satisfied by a cube dragged straight into the platform.
## 0.10 put the cube's underside 35 mm above the support, and the arm could not reach it: measured
## over 681 carries the height reached is a median of 81.8 mm, with only 24% clearing 100 mm. Since
## the drop zone only becomes the goal once the lift is DONE, that gate left three carries in four
## stuck at the lift pose -- the arm picked the cube up, held it, and never set off. 0.08 still clears
## the 50 mm support by 15 mm of cube underside and sits under the median. Then re-measured on the
## next run: the median had fallen to 72.4 mm and only 30% cleared 80. The gate and the height feed
## each other -- the traverse is locked behind the lift, so a gate the arm rarely reaches means it
## rarely gets to practise the rest, and the height it bothers to reach drops further. 0.07 leaves
## 5 mm of cube underside over the support and puts the gate under the median again.
## Distance over which the FINAL approach is paid, in metres.
##
## The pose shaping is linear over `workspace_scale` (0.5 m), so closing the last 33 mm is worth
## 0.066 of the position term and about 0.03 of task progress -- next to nothing against the risk of
## a collision or of shoving the cube. Measured over 500 episodes of v22 the arm stops with a median
## pose error of exactly 33 mm, and every capture condition then misses by a hair: depth 30.4 mm
## against a 30 mm limit, sideways 32.6 against 30, across 10.2 against 10. It is not imprecision,
## it is a reward with no gradient left where the grasp happens.
@export var final_approach_span := 0.05
## Which way round the wrist should be, as the world-up component of the hand's local X axis.
##
## Orientation is out of the task as a GATE -- the claws come down from above and the roll about that
## axis is free -- but free is not the same as arbitrary, and the roll was landing either way up.
## Measured across four seeds the hand's local X reads +0.677, -0.675, +0.647, -0.629: the same pose
## mirrored, motor block up in half of them and under in the other half. This asks for one of the two.
## +1 wants the motor up; flip to -1 if it settles the wrong way round.
@export_range(-1.0, 1.0, 2.0) var wrist_up_sign := 1.0
## Alignment below which the wrist term pays nothing, so it cannot compete with reaching the cube.
@export var wrist_up_deadband := 0.0
## Depth below the lift height that costs the full "carrying it too low" penalty, in metres.
##
## The lift is required before the traverse and paid while it happens, but nothing charged the arm
## for LETTING IT BACK DOWN afterwards -- so it could satisfy the gate, drop to the floor and slide
## the rest of the way, cube scraping the ground or under it. This charges every step the held cube
## sits below the height it was asked to reach.
@export var carry_low_scale := 0.03

## 0.105, non 0.08. Il supporto sotto la zona e' stato rialzato: la sua cima e' passata da 50.0 a
## 74.2 mm, e il cubo consegnato appoggia a 92.3 mm invece di 86.5. Con 80 mm di sollevamento il cubo
## saliva a 95 mm, cioe' 5.8 mm sopra il supporto e 2.7 sopra la quota di consegna -- nessun margine,
## e il trasporto passava radente. 105 lo porta a 120 mm: 30.8 mm sopra il supporto.
@export var carry_lift_height := 0.105

## Quanto il cubo deve essersi staccato dal piano perche' le chele smettano di essere esenti dalle
## collisioni con l'ambiente. Sotto questa quota l'esenzione serve alla presa; sopra, e' solo un
## permesso di attraversare gli ostacoli.
@export var claw_exemption_clearance := 0.03

## Frazione della salita oltre la quale il bersaglio del trasporto comincia a spostarsi verso la
## zona. Sotto questa soglia salire vale, ma non avvicina: serve a impedire la diagonale che passa
## attraverso l'ostacolo accanto alla zona.
@export var carry_cross_after := 0.75
## Over how many metres the return home is scored, from the home pose outwards.
##
## 0.15, not 0.40. Measured: the best v45 policy stops 103 to 118 mm short of home and holds there.
## Over a 400 mm span that position already reads 0.97 of the way back, so the reward had nothing
## left to pull with -- the arm was, correctly, done. Over 150 mm the same position reads 0.25, and
## the last hundred millimetres are worth ten times more per millimetre than they were.
##
## Nothing ever required the arm to finish the return, either: `success` in every number this project
## has printed for the xArm is the `target_reached` event, and the scenario fires that at the
## DELIVERY. "100% success" always meant "delivered", never "delivered and home".
## 0.30, non 0.15. Il ritorno da sopra la zona e' lungo 272 mm: con una banda da 150 la prima meta'
## del viaggio non vale NIENTE -- il progresso resta inchiodato a 0.900, misurato nel run isolato,
## dove 150 episodi su 150 hanno chiuso con progresso 0.900 esatto. Senza gradiente per 122 mm una
## policy non ha modo di incamminarsi. Con 0.30 la banda copre il viaggio vero, e a 105 mm da casa
## legge 0.65, quindi l'ultimo tratto resta distinguibile.
@export var home_return_span := 0.30
## Per-step cost for disturbing the object after it has been delivered.
##
## The job ends with the arm back home and the piece left alone. Without this the delivery pays 1.0
## and the remaining steps are free, so nudging the delivered cube around costs nothing.
@export var delivered_disturbance_scale := 0.005
## Cost for fidgeting with the gripper while carrying, per unit of commanded travel.
##
## Once the cube is in the hand the claws have one job: stay where they are. Any command on that joint
## while carrying is either squeezing harder for nothing or working towards dropping it.
@export var carry_gripper_still_scale := 0.5
## One-off cost for losing the cube after having caught it.
##
## Dropping it outside the zone undoes the whole episode, and until now it cost only the progress that
## went with it -- which the retarget promptly rebased away.
@export var grasp_lost_penalty := 5.0
## The cube counts as delivered once it is inside the footprint and has stopped moving.
@export var drop_settle_speed := 0.05

@export_category("Reverse curriculum")
## Start the episode with the hand already beside the cube, then retreat along the approach axis.
##
## Training v3 ran 4149 episodes and never grasped once: it learned to approach (position error 42 ->
## 17 cm) and stopped there, arriving at a median speed of 0.849 -- fast, not braking -- and knocking
## the cube away in 18% of episodes. Best-in-episode progress never crossed 0.5, the value only a
## capture can pass, so the grasp branch was never sampled and the critic never learned what it is
## worth. Starting AT the cube inverts that: closing is learned first, approaching afterwards.
##
## The rungs sit on the APPROACH AXIS, not on an interpolation of joint angles. Measured on the
## OpenArm, interpolating joints moves the hand 39.8 mm sideways against 17.9 mm along the approach,
## so the return sweeps the object away instead of enclosing it.
@export var reverse_curriculum_enabled := true
## Standoff above the cube at the easiest rung and at the full task, in metres.
@export var reverse_standoff_near := 0.0
## Highest rung of the reverse curriculum, in metres above the cube.
##
## 0.30. The old value was 0.135, set because at 180 mm six of eight seeds came out with the wrist
## rolled under -- and that was measured against the OLD layout's wrist reference, which left 35 to
## 132 degrees of posture error everywhere and so was selecting contorted poses at every height. With
## the reference re-measured the ladder was probed again to 300 mm: IK error 2.9 to 3.0 mm on EVERY
## rung, posture consistent throughout. The ceiling was never the arm's reach.
##
## Why it matters more than it looks: the top rung is where a policy's training ends, and 135 mm was
## not the task. Measured on the v45 best checkpoint, 100% of frozen evaluations succeeded through
## the curriculum and only 8 of 20 succeeded FROM HOME. The ladder has to reach far enough that the
## last rung and the home start are the same problem.
@export var reverse_standoff_far := 0.30
## Share of episodes that skip the ladder entirely and start at the home pose.
##
## The rungs are an assist, and no rung is the real start: home is 43 cm from the cube. Extrapolating
## from the top rung is what the 100%-through-the-curriculum / 40%-from-home gap is made of. A slice
## of genuine home starts trains the thing that is actually asked for, and mixing it in avoids the
## cliff that retiring the assist outright caused on v17 -- there the task changed under the policy in
## one step, here both starts are in the same distribution.
@export_range(0.0, 1.0, 0.05) var home_start_fraction := 0.2
## The retreat is earned, not scheduled. Advancing on EPISODE COUNT was the first version and it
## failed: linear over 6000 episodes put the standoff at 15 mm by episode 500 and 45 mm by 1500, so
## the policy was pushed off the easy rung long before it could use it -- 483 episodes, zero
## captures, with the cube knocked away in 27-63% of them. Measured separately, closing the claws
## from the standoff-0 seed DOES capture (34 steps), so the rung works; it was being withdrawn.
##
## Each capture moves the start a little further back; failures never move it forward. No demotion:
## on the OpenArm, retreating on failure made the policy forget whole regions, with the signature of
## failures spread evenly instead of concentrated on the hard ones.
## How much of the retreat one credited capture buys.
##
## 0.02 of a 0.18 m span is a ninth of the whole curriculum per capture: v17 went from the seeded
## start to fully retreated in NINE captures, reached by episode 47, before anything could consolidate.
## 0.002 gives ninety, which is the difference between a curriculum and a trapdoor.
@export var reverse_advance_per_capture := 0.002
## Kept for the probe and for reporting: the episode by which a purely scheduled retreat would end.
@export var reverse_full_episode := 6000
## 32, not 8: the seed has to satisfy three things at once now -- reach the rung, hold the shared
## wrist posture, and land on the branch that turns anticlockwise -- and with eight tries a couple of
## spawns in twenty-five found no such pose and fell back to whichever was closest, which is where the
## clockwise starts came from. Tries cost only reset time.
##
## How close the solve has to get before the rung is used at all. Beyond this the episode starts
## from home instead: an inaccurate seed is not a gentler task, it is a different one.
@export var reverse_seed_tolerance := 0.012
## How many starting configurations the seeding solver may try before giving up on the rung.
## How far down the hand must point for a seeded rung to be preferred, as the vertical component of
## the approach axis (-1 straight down, 0 horizontal).
##
## The seeding IK does not track orientation -- orientation is out of this task -- and it randomises
## its start across the whole joint range, so it accepts ANY branch that puts the grasp point on the
## cube, wrist rolled under included. Measured over 40 seeds: 20% came out with the hand not pointing
## down at all, the worst at -0.054, essentially horizontal, and the arm started the episode with its
## wrist motor below the cube. It still grasps from there, but it is a different task from the one the
## other 80% see, out of the same rung.
@export var reverse_seed_min_downward := 0.5
## The wrist posture every seeded grasp should share, as the hand's X and Y axes IN THE BASE'S FRAME.
##
## Asked for: one hand orientation for all spawns, the one it has when the cube is on the right --
## not two mirrored postures. Expressed in the base frame because the base rotation is exactly what
## SHOULD differ between spawns; taking it out leaves only the posture, which then reads the same
## however far round the arm has swung.
##
## Measured across the spawn area on the seeded grasp: four of five positions already sit at
## x=(0.0..0.43, +0.55, -0.8), y=(0, -0.83, -0.55), and the fifth -- the far left -- drifts to
## x=(-0.32, +0.27, -0.91). These defaults are the right-hand-side reading.
## Measured on the new layout, not carried over: with the areas moved the seeding samples cluster on
## one posture -- the hand's own +Y within a few degrees of straight DOWN (which IS the black motor
## block being up) and its +X along the direction the arm reaches in, base-frame +Z. The old constants
## were a good pose from the old layout and left 35 to 132 degrees of error here, so the filter they
## feed refused every correct pose in silence. These are the ideal form of the measured cluster.
@export var wrist_reference_x := Vector3(0.0, 0.0, 1.0)
@export var wrist_reference_y := Vector3(0.0, -1.0, 0.0)
## Dove deve puntare la linea fra le due dita, nel frame della base.
##
## Misurato sulla posa di CASA -- la mano a casa ha z = (1, 0, 0) nel frame della base -- e non dal
## grappolo delle pose seminate: un riferimento ricavato da cio' che si sta giudicando e' d'accordo
## con se' stesso anche quando e' storto.
@export var wrist_reference_fingers := Vector3(1.0, 0.0, 0.0)
## I link delle chele, tenuti QUI perche' l'editor svuota gli array sul nodo del braccio a ogni
## salvataggio. Vedi `_restore_claw_links`.
@export var claw_link_names := PackedStringArray(["finger_left_link", "finger_right_link"])
## How far a seeded posture may sit from that reference before it is refused, in degrees.
@export var wrist_reference_tolerance_degrees := 35.0
@export var reverse_seed_attempts := 32
## IK iterations spent placing the hand on the rung. Runs synchronously inside the reset: kinematic
## control applies a joint write immediately, so no physics frame is needed between iterations.
@export var reverse_solve_iterations := 500

@export_category("Orientation")
## Whether the demanded wrist orientation is part of the task at all.
##
## False for this scene: the claws come down on the cube from ABOVE and cannot approach it in line,
## so the roll about the approach axis is free -- the fingers close on whichever two faces they meet.
## Demanding a full orientation anyway made the task unsatisfiable: measured with IK, 8 spawns out of
## 9 sat outside the 25 deg gate even with the claws already on the cube.
##
## The orientation error stays in the OBSERVATION either way. Removing it there would change the
## 29-value contract and invalidate every checkpoint; it costs nothing to leave the policy able to
## see an angle it is no longer scored on.
@export var orientation_matters := false

@export_category("Pose gates")
@export var success_distance := 0.03
@export_range(0.1, 180.0, 0.1) var success_angle_degrees := 25.0
@export var success_hold_physics_frames := 20
@export var success_max_joint_speed := 0.30
@export var joint_jitter_degrees := 0.0

@onready var controller: ScenarioController = $ScenarioController
@onready var arm = $RobotArm
@onready var goal_event = $ScenarioController/ScenarioEventSystem/GoalReached
## The delivery has its own event now, so `target_reached` can mean what every metric in the project
## reads it as: the task is over. It used to be fired at the DELIVERY, which is why "100% success"
## meant "delivered" and the return home was neither required nor measured -- three times now a
## number here has been about an earlier link in the chain than the task.
@onready var delivered_event = get_node_or_null(
	"ScenarioController/ScenarioEventSystem/Delivered")
@onready var collision_event = $ScenarioController/ScenarioEventSystem/Collision
## Picking the cube up is a MILESTONE, and until now only delivering it was paid as one.
##
## Measured from the seeded rung over 120 steps: standing still returns 7.01 and capturing returns
## 2.76. Capturing switches the goal to the drop pose, which is far away, so every dense near-target
## term collapses -- parking on the cube farms them and grasping loses them. The policy was right and
## the reward was wrong. This pays for the act itself, independent of the shaping.
@onready var grasp_event = $ScenarioController/ScenarioEventSystem/Grasped
@onready var lost_event = get_node_or_null("ScenarioController/ScenarioEventSystem/Lost")

var _training_episode := 0
var _training_mode := false
var _goal_terminal_reason := "target_reached"
var _episode_rng := RandomNumberGenerator.new()
## World basis the grasp pose is held at for the whole episode, so knocking the object askew does
## not drag the goal along with it.
var _demanded_basis := Basis.IDENTITY
var _demanded_basis_valid := false
## Where the object was set down. The disturbance penalty is measured from here.
var _spawn_position := Vector3.ZERO
var _spawn_override := Vector3.ZERO
var _spawn_override_active := false
var _grasp_attach_transform := Transform3D.IDENTITY
var _grasp_frozen_before := false
var _grasp_gripper_position := 0.0
var _grasp_gripper_held := false
var _object_half_extents := Vector3.ZERO
var _surface_top_y := 0.0
var _floor_breached := false
## Il cubo ha attraversato il varco della zona almeno una volta: e' la consegna.
var _zone_crossed := false
## Where the cube was, relative to the zone centre, at the frame it crossed. Logged and paid once.
var _crossing_offset := Vector2.ZERO
var _crossing_reward_pending := 0.0
## Le chele restano esenti finche' il cubo non e' sollevato: da li' in poi sono link come gli altri.
var _claw_exempt_links: PackedStringArray = PackedStringArray()
var _ledge_top := -INF
var _ledge_centre := Vector3.ZERO
var _ledge_half := Vector3.ZERO
var _breach_depth_max := 0.0
var _breach_source := ""
var _calibrated := false
## [[claw link, tip position in that link's frame], ...] filled once at calibration. The tool point
## is their live midpoint, so it travels with the claws.
var _finger_tips: Array = []
## Drop phase state. `_carrying` spans capture to release; `_delivered` latches for the episode so
## the goal cannot be collected twice.
var _carrying := false
var _delivered := false
var _drop_pose: Marker3D = null
## Where the tool sits with the arm at its home joints. The last leg of the task is getting back here
## after the release, so it needs a pose the arm can be pointed at like any other.
var _home_pose: Marker3D = null
## Straight up from where the cube was caught. The arm is sent HERE first and only pointed at the drop
## zone once the lift is done, so the two legs happen in order instead of being traded off.
var _lift_pose: Marker3D = null
var _lifted := false
var _carry_from := Vector3.ZERO
## IK used only to place the hand on a reverse-curriculum rung at reset. Kept out of the episode:
## it never runs while the policy is acting.
var _reverse_ik: URDFIKController = null
var _reverse_target: Marker3D = null
## How far the reverse curriculum has retreated, 0 at the cube and 1 at the full task. Earned by
## captures, never by elapsed episodes.
var _reverse_level := 0.0
## Rungs the solver could not reach, so the run can be judged on how often the assist actually
## applied rather than on the assumption that it always does.
var _reverse_seed_failures := 0
## True when this episode skipped the ladder and started from the home pose.
var _home_start_episode := false
var _episode_log_warned := false
## Closest the tool got to home AFTER the delivery, in metres.
var _home_closest := INF
var _home_offset_at_closest := [0.0, 0.0, 0.0]
var _home_joint_errors: Array = []
var _home_pose_best := INF
var _capture_across := -1.0
var _capture_gap := -1.0
var _capture_depth := -1.0
var _capture_side := -1.0
var _capture_outside := -1.0
var _release_tip_tilt := -1.0
var _release_axis_pitch := -1.0
var _home_speed_at_closest := 0.0
## Distance to the demanded target at the instant the agent's own gate declared success (-1: never).
var _gate_success_distance := -1.0
var _gate_success_target := ""
## Where the cube ends up while nobody is holding it. Both numbers answer a specific doubt raised
## from watching a render: that the claws swat the cube away instead of closing on it, and that they
## sometimes drive it clean through the floor. The floor is a static box and the cube a small rigid
## body with continuous detection off by default, so tunnelling is not a far-fetched worry.
var _cube_lowest := INF
var _cube_max_shift := 0.0
var _cube_spawn := Vector3.ZERO
## How far the cube has been turned from the pose it was placed in. A knocked cube that yaws is not
## a cosmetic problem: a 30 mm cube presents a 42 mm diagonal at 45 degrees, so the claws that fit
## around it at rest no longer do, and the gap test starts refusing grasps for a reason that has
## nothing to do with the policy.
var _cube_max_turn := 0.0
var _cube_spawn_basis := Basis.IDENTITY
var _cube_last_seen := Vector3.ZERO
var _final_approach_last := INF
## Deepest the claw TIPS have gone below the surface. The claws are exempt from the floor collision
## on purpose -- a cube resting on the ground cannot be grasped by claws forbidden to reach it -- so
## nothing at all stops them going arbitrarily deep, and nothing measured how deep they went.
var _tip_lowest := INF
## Highest the cube got while actually held. Dragging it to the zone along the floor satisfies both
## the carrying progress (which only measures distance to the drop pose) and the delivery test (which
## has no vertical bound), so nothing so far has asked for a lift.
var _carry_highest := -INF
## What the policy actually asks the gripper for, and what the joint actually does about it. Kept
## side by side on purpose: "the claws never close" has two completely different causes -- a policy
## that never commands a closure, or a command that never reaches the joint -- and the capture
## counters cannot tell them apart.
var _grip_command_min := INF
var _grip_command_sum := 0.0
var _grip_command_count := 0
var _grip_closure_max := 0.0
## The tool point, held rigid in the hand instead of riding the claw tips. See _calibrate_finger_tips.
var _tool_offset_frozen := Vector3.ZERO
var _tool_offset_frozen_valid := false
var _goal_paid := false
var _cube_pin_left := 0
var _cube_pin_pushes := 0
var _cube_speed_clamps := 0
var _cube_rescues := 0
## One rung of credit per episode, however many times the cube is picked up in it.
var _credited_capture := false
## Why the capture was refused, counted per episode. Four blind levers were tried before this was
## measured; each rejection reason needs a different fix, so guessing between them is expensive.
@export var log_capture_rejections := true
var _capture_attempts := 0
var _reject_across := 0
var _reject_between := 0
var _reject_reach := 0
var _reject_depth := 0
var _reject_side := 0
var _reject_gap := 0
var _reject_speed := 0
var _best_across := INF
var _best_between := INF
var _best_reach := INF
var _best_depth := INF
var _best_side := INF
var _best_gap := INF
var _best_speed := INF
## How much of each claw counts as its tip, measured back from its deepest point.
const TIP_BAND := 0.005


func _ready() -> void:
	var configured_terminal_reason := str(goal_event.terminal_reason)
	if not configured_terminal_reason.is_empty():
		_goal_terminal_reason = configured_terminal_reason
	if grasp_target == null:
		grasp_target = get_node_or_null("RigidTarget") as RigidBody3D
	if grasp_target_pose == null and grasp_target != null:
		grasp_target_pose = grasp_target.get_node_or_null("GraspPose") as Marker3D
	if grasp_target == null or grasp_target_pose == null:
		push_error("xarm_grasp_zone_scenario needs a RigidBody3D target with a GraspPose marker")
		return
	if spawn_area == null:
		spawn_area = get_node_or_null("SpawnArea") as Area3D
	if drop_zone == null:
		drop_zone = get_node_or_null("DropZone") as Area3D
	# Its own node, not a child of the zone: the demanded pose has to sit ABOVE the footprint, and a
	# marker parented to the zone would inherit any later move of it.
	_drop_pose = Marker3D.new()
	_drop_pose.name = "DropPose"
	add_child(_drop_pose)

	_home_pose = Marker3D.new()
	_home_pose.name = "HomePose"
	add_child(_home_pose)

	_lift_pose = Marker3D.new()
	_lift_pose.name = "LiftPose"
	add_child(_lift_pose)

	arm.target = grasp_target
	arm.target_pose = grasp_target_pose
	# The grasp geometry asks where the OBJECT is, which moves; the pose above is what is DEMANDED.
	arm.grasp_object_pose = grasp_target_pose
	arm.target_reached.connect(_on_target_reached)
	arm.target_pose_relocated.connect(_on_target_pose_relocated)
	arm.obstacle_collision.connect(_on_obstacle_collision)
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_on_episode_reset_started)
	_restore_claw_links()
	_calibrate()
	_calibrate_finger_tips()
	_claw_exempt_links = arm.environment_collision_exempt_links.duplicate()
	_measure_ledge()
	_capture_home_pose()
	_build_reverse_ik()


func _physics_process(_delta: float) -> void:
	# Re-pin every frame rather than only at reset: the object is a rigid body, so a graze rotates
	# it, and a demanded pose parented to it would rotate too -- moving the goal exactly in the
	# episodes where the arm has already made a mistake.
	if _demanded_basis_valid and grasp_target_pose != null and not arm.is_grasp_attached():
		grasp_target_pose.global_basis = _demanded_basis
	# Success means the cube is DELIVERED, not that the arm got there holding it. Without this the
	# agent's own gates were satisfied by carrying the cube over the zone and keeping hold: 95% success
	# and not one delivery, and `--best-checkpoint` was selecting exactly that behaviour.
	if drop_enabled:
		# Consegnato NON basta: finche' la posa non e' quella di casa, il compito non e' finito. Il
		# gate di posizione da solo si accontenta di un utensile nel punto giusto con il braccio in
		# una configurazione qualunque.
		arm.success_blocked = (not _delivered
			or (home_return_required
				and _home_joint_error() > deg_to_rad(maxf(home_pose_tolerance_degrees, 0.1))))
	# L'esenzione delle chele esiste per la PRESA: devono scendere accanto al cubo, e il pavimento
	# che deve fermare un avambraccio non deve fermare una punta. Finita la salita non serve piu', e
	# lasciarla accesa e' quello che permette di strisciare sulla sporgenza col cubo in mano senza
	# che succeda niente.
	# L'esenzione vale solo mentre si va a PRENDERE: serve perche' le punte devono scendere accanto al
	# cubo, dove il pavimento che ferma un avambraccio non deve fermare una punta. Dopo la consegna
	# non c'e' piu' niente da prendere, e lasciarla accesa e' quello che permetteva alle chele di
	# entrare dentro il supporto senza che l'episodio finisse -- visto a schermo.
	# Armate appena il cubo si stacca dal piano, non a salita FINITA.
	#
	# Con l'esenzione legata a `_lifted` le chele restavano libere per tutta l'ascesa, che e'
	# esattamente la fase in cui passano accanto alla sporgenza: su una scena con un ostacolo alto
	# 11 cm accanto alla zona, le chele lo attraversavano e trenta episodi di seguito riportavano
	# ZERO collisioni. Il difetto non era il rilevamento -- era che la finestra di esenzione durava
	# fino a 14.5 cm, sopra l'ostacolo.
	#
	# L'esenzione serve solo mentre si va a PRENDERE, quando le punte devono scendere accanto al cubo
	# e il pavimento che ferma un avambraccio non deve fermare una punta. Bastano pochi centimetri di
	# distacco perche' non serva piu'.
	var clear_of_surface := (grasp_target != null
		and grasp_target.global_position.y - (_surface_top_y + _object_half_extents.y)
			> claw_exemption_clearance)
	arm.environment_collision_exempt_links = (
		PackedStringArray() if (_lifted or _delivered or (_carrying and clear_of_surface))
		else _claw_exempt_links)
	# The whole chain, latched from facts this scenario owns.
	#
	# Two attempts at this were wrong before it was: reading the agent's has_succeeded() at the next
	# reset (already cleared), then reading it here (the agent sets it later in the same frame, and
	# terminates, so it is never seen). Both reported zero successes for episodes that had finished.
	# What is not ambiguous is the geometry: the cube is delivered, and the tool came back inside the
	# same home gate the reward pays for.
	# Il MIGLIOR errore di posa dell'episodio, a parte. Quello registrato al punto piu' vicino a casa
	# raccontava un'altra storia: il braccio puo' passare per la posa di casa in un istante e poi
	# avvicinarsi di piu' al PUNTO con una configurazione diversa, e leggendo solo il secondo istante
	# la base risultava a 103 gradi anche negli episodi che il successo lo avevano preso.
	if _delivered:
		_home_pose_best = minf(_home_pose_best, _home_joint_error())
	if _delivered and _home_pose != null and arm.tool_pose != null:
		var away := (arm.tool_pose as Node3D).global_position - _home_pose.global_position
		if away.length() < _home_closest:
			_home_closest = away.length()
			# Per asse, e con la velocita' dei giunti in quel momento: un braccio che si ferma perche'
			# non puo' andare oltre e uno che si ferma perche' ha smesso di provarci hanno lo stesso
			# aspetto in una distanza sola.
			_home_offset_at_closest = [away.x * 1000.0, away.y * 1000.0, away.z * 1000.0]
			_home_speed_at_closest = float(arm.call("_max_joint_speed"))
			# Per GIUNTO, non solo la media: "torna girato male" e "torna col gomito altrove" hanno
			# la stessa media e fix diversi.
			_home_joint_errors = []
			var robot_now := arm.get_node_or_null("xarm")
			var home_now: Dictionary = arm.call("_build_home_positions")
			if robot_now != null:
				for joint_name in arm.get_controlled_joint_names():
					_home_joint_errors.append(rad_to_deg(
						robot_now.get_joint_position(String(joint_name))
						- float(home_now.get(String(joint_name), 0.0))))
	# And where the arm actually WAS when its own gate declared success, because the two disagree:
	# the gate reported success on episodes whose tool never came within 100 mm of home, and which of
	# the two is measuring the task has to be settled by looking, not by preferring one.
	if _gate_success_distance < 0.0 and bool(arm.call("has_succeeded")):
		_gate_success_distance = 0.0
		if arm.tool_pose != null and arm.target != null:
			_gate_success_distance = (arm.tool_pose as Node3D).global_position.distance_to(
				(arm.target as Node3D).global_position)
			_gate_success_target = String((arm.target as Node3D).name)
	_sync_tool_pose_to_tips()
	_update_grasp()
	_track_loose_cube()
	_track_gripper()
	if grasp_target != null and arm.is_grasp_attached():
		_carry_highest = maxf(_carry_highest, grasp_target.global_position.y)
	if _finger_tips.size() >= 2:
		_tip_lowest = minf(_tip_lowest, minf(_tip_world(0).y, _tip_world(1).y))
	_check_floor_breach()
	_pin_cube_after_reset()
	_tame_cube()


## Holds the cube on its spawn spot while the arm settles into the seeded pose.
func _pin_cube_after_reset() -> void:
	if _cube_pin_left <= 0 or grasp_target == null:
		return
	_cube_pin_left -= 1
	if arm.is_grasp_attached():
		_cube_pin_left = 0
		return
	var drift := grasp_target.global_position.distance_to(_cube_spawn)
	if drift > 0.002:
		_cube_pin_pushes += 1
	grasp_target.global_position = _cube_spawn
	grasp_target.linear_velocity = Vector3.ZERO
	grasp_target.angular_velocity = Vector3.ZERO


## Ends the episode when the claws -- or the object they are holding -- go THROUGH the ground.
##
## Two separate holes made this possible, and neither was covered by a reward term:
##
##   the claws       are listed in environment_collision_exempt_links, and that exemption is
##                   unconditional: urdf_robot_arm_agent.gd skips the shape query for those links at
##                   ANY depth. The exemption is still needed -- a cube resting on the floor cannot
##                   be grasped by claws forbidden to reach it -- so what is added here is a DEPTH
##                   BUDGET, not the old blanket rule that made the exemption necessary.
##
##   the held object is never checked at all: it is not part of the robot, so it is absent from
##                   _robot_collision_shapes, and _tame_cube's floor rescue deliberately skips while
##                   it is attached. Once captured it is frozen kinematic and written from the hand
##                   every frame, so it passes through the floor with nothing resisting and nothing
##                   reporting.
##
## Measured against the tips, which is what the graded penalty uses, so cost and terminal agree on
## what "under" means instead of disagreeing at the boundary.
## Legge il blocco che regge la zona: cima e impronta, una volta, dalla scena.
func _measure_ledge() -> void:
	var body := get_node_or_null("SupportDropZone") as Node3D
	if body == null:
		return
	for child in body.get_children():
		var shape := child as CollisionShape3D
		if shape == null or not (shape.shape is BoxShape3D):
			continue
		# Le mezze estensioni vanno prese NEL MONDO: il nodo della forma puo' avere una scala, e
		# `size * 0.5` da solo legge le sue unita' come se fossero metri. Misurato dopo che l'utente
		# ha rifatto il supporto: cima a 46 METRI e impronta 132 x 28, cioe' una regola che avrebbe
		# dichiarato ogni trasporto uno sfondamento.
		var half: Vector3 = (shape.shape as BoxShape3D).size * 0.5 * shape.global_transform.basis.get_scale()
		if shape.global_position.y + half.y > _ledge_top:
			_ledge_top = shape.global_position.y + half.y
			_ledge_centre = shape.global_position
			_ledge_half = half
	if _ledge_top > -INF:
		print("[sporgenza] cima a %.4f, impronta %s" % [_ledge_top, str(_ledge_half)])


## Il cubo in mano sta attraversando la sporgenza?
##
## Solo dentro l'impronta del blocco: fuori di li' non c'e' niente da attraversare, e un limite di
## altezza applicato ovunque vieterebbe il trasporto basso che non tocca nulla.
func _held_object_through_ledge() -> bool:
	if _ledge_top <= -INF or grasp_target == null or not arm.is_grasp_attached():
		return false
	var offset := grasp_target.global_position - _ledge_centre
	if absf(offset.x) > _ledge_half.x or absf(offset.z) > _ledge_half.z:
		return false
	return (grasp_target.global_position.y - _object_half_extents.y
		< _ledge_top - maxf(ledge_breach_object_depth, 0.0))


func _check_floor_breach() -> void:
	if _floor_breached or arm == null or not _calibrated:
		return
	var source := ""
	var lowest := INF
	var limit := INF
	if _finger_tips.size() >= 2:
		lowest = minf(_tip_world(0).y, _tip_world(1).y)
		limit = _surface_top_y - _allowed_claw_depth()
		if lowest < limit:
			source = "claws"
	if source.is_empty() and grasp_target != null and arm.is_grasp_attached():
		# The cube is a box, so its lowest point is the centre less the half extent.
		var object_lowest: float = grasp_target.global_position.y - _object_half_extents.y
		var object_limit := _surface_top_y - maxf(floor_breach_object_depth, 0.0)
		if object_lowest < object_limit:
			lowest = object_lowest
			limit = object_limit
			source = "held_object"
	if source.is_empty() and ledge_breach_terminates and _held_object_through_ledge():
		lowest = grasp_target.global_position.y - _object_half_extents.y
		limit = _ledge_top - maxf(ledge_breach_object_depth, 0.0)
		source = "ledge_held_object"
	if source.is_empty():
		return
	_floor_breached = true
	_breach_source = source
	_breach_depth_max = _surface_top_y - lowest
	print("[pavimento] SFONDAMENTO da %s: %.1f mm sotto il piano (concessi %.1f)" % [
		source, _breach_depth_max * 1000.0, (_surface_top_y - limit) * 1000.0])
	if floor_breach_terminates:
		arm.report_obstacle_collision("floor_breach_%s" % source)


## How far under the surface the claws may currently go: deep while a grasp is plausible, shallow
## otherwise. Buying the depth with proximity is what lets the rule forbid digging without forbidding
## the grasp -- see floor_breach_depth for the measurements that ruled out a plain depth limit.
func _allowed_claw_depth() -> float:
	var shallow := maxf(floor_breach_depth, 0.0)
	if grasp_target == null or _finger_tips.size() < 2:
		return shallow
	var centre := (_tip_world(0) + _tip_world(1)) * 0.5
	if centre.distance_to(grasp_target.global_position) > maxf(floor_breach_grasp_radius, 0.0):
		return shallow
	return maxf(floor_breach_grasp_depth, shallow)


## Keeps the loose cube inside the world: speed capped, never below the surface it rests on.
func _tame_cube() -> void:
	# Also skipped once the cube has been let go over the zone: the release is a deliberate drop and
	# capping or rescuing it there is the scenario fighting its own delivery.
	if grasp_target == null or arm.is_grasp_attached() or _carrying or _delivered:
		return
	var speed := grasp_target.linear_velocity.length()
	if cube_max_speed > 0.0 and speed > cube_max_speed:
		grasp_target.linear_velocity = (
			grasp_target.linear_velocity / speed * cube_max_speed)
		_cube_speed_clamps += 1
	if not cube_floor_rescue:
		return
	# The trigger is the cube's CENTRE reaching the surface plane, not the cube being pressed a
	# little: a legitimate grasp squashes it a few millimetres down every time (measured, it bottoms
	# out around 1 mm above the floor), and rescuing at that depth fights the claws and cost the
	# scripted run every one of its three grasps.
	if grasp_target.global_position.y >= _surface_top_y:
		return
	# Centre under the floor. Nothing above ground can put it there, so this is the solver having
	# shoved it through; put it back on the surface and stop it dead.
	var resting := _surface_top_y + _object_half_extents.y
	var rescued := grasp_target.global_position
	rescued.y = resting
	grasp_target.global_position = rescued
	grasp_target.linear_velocity = Vector3.ZERO
	_cube_rescues += 1


## The gripper command the agent last applied, and how far the claws have actually travelled.
func _track_gripper() -> void:
	var index := Array(arm.get_controlled_joint_names()).find(String(arm.gripper_joint_name))
	if index < 0:
		return
	var applied: Array = arm.get_previous_action_observation()
	if index >= applied.size():
		return
	var command := float(applied[index])
	_grip_command_min = minf(_grip_command_min, command)
	_grip_command_sum += command
	_grip_command_count += 1
	_grip_closure_max = maxf(_grip_closure_max, _gripper_closure())


## Signed offset of the cube along the claws' closing axis, for probes.
##
## The capture rule's own "across" test allows less and less of this as the claws shut -- the room to
## spare is (gap - width) / 2 -- so an offset that is harmless while open refuses the grasp at the
## exact moment it completes.
func _across_offset_for_probe() -> float:
	if _finger_tips.size() < 2 or grasp_target == null:
		return NAN
	var left := _tip_world(0)
	var right := _tip_world(1)
	var span := left - right
	if span.length() < 0.0001:
		return NAN
	return (grasp_target.global_position - (left + right) * 0.5).dot(span.normalized())


## Records how far the cube gets knocked while it is loose, and how far under the floor it goes.
func _track_loose_cube() -> void:
	# Only before the first capture. Afterwards the cube is carried and then deliberately let go over
	# the drop zone, and counting that as "knocked about" would report every clean delivery as the
	# worst shove of the episode.
	if grasp_target == null or arm.is_grasp_attached() or _credited_capture:
		return
	var here := grasp_target.global_position
	_cube_lowest = minf(_cube_lowest, here.y)
	_cube_max_turn = maxf(_cube_max_turn, rad_to_deg(
		grasp_target.global_basis.get_rotation_quaternion().angle_to(
			_cube_spawn_basis.get_rotation_quaternion())))
	_cube_max_shift = maxf(_cube_max_shift, (here - _cube_spawn).length())


## Puts the tool frame on the live midpoint between the CLAW tips, so it travels with them.
##
## Only the POSITION is moved. The orientation stays whatever the wrist presents, because that is a
## separate question and moving both at once would make either fix impossible to attribute.
##
## Measured on the real claws, the midpoint runs y +0.1153 (shut) to +0.0863 (open) in the hand
## frame -- 28.9 mm of travel that always stays between the claws. An earlier version of this froze
## the point instead, on the argument that the midpoint retreated towards the wrist when opening;
## that was an artifact of measuring `grip_left/right`, the linkage arms, which stop 50 mm short of
## the claws.
##
## The offset is written in the tool node's PARENT frame (it hangs off EndEffector, which the agent
## re-poses from forward kinematics each frame), so a one-frame lag in the parent cannot drag it off.
func _sync_tool_pose_to_tips() -> void:
	if not tool_pose_follows_finger_tips or _finger_tips.size() < 2:
		return
	var tool: Node3D = arm.tool_pose
	if tool == null:
		return
	var parent := tool.get_parent() as Node3D
	if parent == null:
		return
	var robot := arm.get_node_or_null("xarm")
	if robot == null:
		return
	var hand := robot.get_link_node(String(arm.tcp_link_name)) as Node3D
	if hand == null:
		return
	if _finger_tips.size() < 2:
		return
	var point := (
		hand.global_transform * _tool_offset_frozen
		if _tool_offset_frozen_valid
		else _grasp_point_world())
	tool.position = parent.global_transform.affine_inverse() * point


## World position of one claw tip, from the collision vertex cached at calibration time.
func _tip_world(index: int) -> Vector3:
	var entry: Array = _finger_tips[index]
	var link: Node3D = entry[0]
	return link.global_transform * (entry[1] as Vector3)


## Finds each finger's tip once, as a MATERIAL POINT kept in that finger's own frame, so the
## per-frame update is a single transform and the point stays the tip at every joint angle.
##
## Picked as the vertex reaching furthest along the hand's own axis, and picked with the gripper
## CLOSED, where the fingers are parallel and "furthest along the hand" is unambiguous. Choosing the
## vertex farthest from the finger's origin instead looks equivalent and is not: splayed open, the
## winning vertex is the one that swung out sideways, and the midpoint of the two drops back towards
## the finger roots -- the tool point then sits in the middle of the HAND rather than between the
## finger tips, which is exactly the wrong place to be while approaching an object.
func _calibrate_finger_tips() -> void:
	_finger_tips.clear()
	if not tool_pose_follows_finger_tips:
		return
	var robot := arm.get_node_or_null("xarm")
	if robot == null:
		return
	var hand := robot.get_link_node(String(arm.tcp_link_name)) as Node3D
	if hand == null:
		return

	# Close the gripper for the measurement, then put it back exactly as it was.
	var gripper := String(arm.gripper_joint_name)
	var joint_node = robot.get_joint_node(gripper)
	var restore_angle: float = robot.get_joint_position(gripper)
	if joint_node != null and joint_node.joint != null and joint_node.joint.limit != null:
		robot.reset_joint_positions({gripper: float(joint_node.joint.limit.lower)})

	for finger_name in arm.gripper_finger_link_names:
		var link := robot.get_link_node(String(finger_name)) as Node3D
		if link == null:
			continue
		# Two passes: find how far this claw reaches, then average every vertex within TIP_BAND of
		# that. A single winning vertex is not mirror-symmetric between the two hulls -- their vertex
		# layouts differ -- so the midpoint of two such vertices sat 3.3 mm off the plane the claws
		# close about, and the tool point looked visibly off-axis.
		var deepest := -INF
		var vertices: Array[Vector3] = []
		for node in _descendants(link):
			var collider := node as CollisionShape3D
			if collider == null or collider.shape == null:
				continue
			for point in _shape_corners(collider.shape):
				var world: Vector3 = collider.global_transform * point
				vertices.append(world)
				deepest = maxf(deepest, (hand.global_transform.affine_inverse() * world).y)
		if deepest <= -INF:
			continue
		var sum := Vector3.ZERO
		var count := 0
		for world in vertices:
			if (hand.global_transform.affine_inverse() * world).y >= deepest - TIP_BAND:
				sum += world
				count += 1
		if count > 0:
			_finger_tips.append([link, link.global_transform.affine_inverse() * (sum / count)])

	# Freeze the tool point in the HAND's frame, sampled at the opening the claws hold the cube in.
	#
	# It used to track the live tip midpoint, and the tips advance 28.9 mm as the claws shut -- so
	# closing the gripper MOVED the point the whole reward is measured from. Measured on a seeded
	# grasp: going from 0.17 to 0.43 closed pushed the pose error from 4.9 mm to 15.2 mm and dropped
	# progress from 0.446 to 0.436. Every step of every closure paid a penalty, against a single
	# uncertain +15 for capturing, and the policy did the sensible thing and kept the hand open --
	# commanding -0.98 for one step and undoing it, 300 times an episode, for eleven versions.
	#
	# The capture rule is unaffected: it measures the cube against the CLAWS, never against this
	# point. Same trap as the reach test and the pad window before it -- a frame that moves with the
	# thing being measured.
	if joint_node != null and joint_node.joint != null and joint_node.joint.limit != null:
		var open_angle := float(joint_node.joint.limit.upper)
		var shut_angle := float(joint_node.joint.limit.lower)
		robot.reset_joint_positions({
			gripper: lerpf(open_angle, shut_angle, clampf(tool_frozen_closure, 0.0, 1.0))})
		_tool_offset_frozen = hand.global_transform.affine_inverse() * _grasp_point_world()
		_tool_offset_frozen_valid = true

	robot.reset_joint_positions({gripper: restore_angle})

	# The agent measured its grasp window during _ready(), against the tool frame as it was THEN --
	# a fixed offset 78 mm short of the claws. Now that the tool sits on the claw tips, that window
	# describes a volume 45 mm away from the tool and no object can ever be inside it. Re-measuring
	# rebuilds it in the corrected frame. Calls the shared agent, changes nothing in it.
	#
	# The tool has to be MOVED first: this runs in _ready(), and the tool only reaches the claws on
	# the first _physics_process, so recalibrating straight away would just re-measure the old frame
	# and print an identical window.
	_sync_tool_pose_to_tips()
	if arm.has_method("_calibrate_gripper_geometry"):
		arm.call("_calibrate_gripper_geometry")
	if _finger_tips.size() < 2:
		push_warning(
			"xarm_grasp_zone: claw tips not found, the tool frame stays at its fixed offset")
		return
	var left := _tip_world(0)
	var right := _tip_world(1)
	var centre := (left + right) * 0.5
	print("[xarm_grasp_zone] tool frame sul punto medio delle punte | scarto dalle due chele "
		+ "%.2f / %.2f mm | punto nel frame mano %s" % [
			centre.distance_to(left) * 1000.0, centre.distance_to(right) * 1000.0,
			hand.global_transform.affine_inverse() * centre])


func _shape_corners(shape: Shape3D) -> PackedVector3Array:
	if shape is BoxShape3D:
		var half: Vector3 = (shape as BoxShape3D).size * 0.5
		var points := PackedVector3Array()
		for sx in [-1.0, 1.0]:
			for sy in [-1.0, 1.0]:
				for sz in [-1.0, 1.0]:
					points.append(Vector3(half.x * sx, half.y * sy, half.z * sz))
		return points
	if shape is ConvexPolygonShape3D:
		return (shape as ConvexPolygonShape3D).points
	return PackedVector3Array()


func _descendants(node: Node) -> Array:
	var out := [node]
	for child in node.get_children():
		out.append_array(_descendants(child))
	return out


## Assisted grasp, mirroring the OpenArm scenario.
##
## The arm runs in KINEMATIC mode, where every link is re-posed from forward kinematics each frame:
## the fingers can push a rigid body but never receive a reaction, so friction can never hold
## anything. Holding therefore has to be declared. Everything about WHEN is measured from the real
## pad faces, so swapping to a frictional grasp under PHYSICS_MOTORS means replacing this one test.
func _update_grasp() -> void:
	if grasp_target == null or not arm.has_method("is_grasp_attached"):
		return
	if arm.is_grasp_attached():
		# The moment the lift is achieved the goal becomes the drop zone. Once only: re-arming it
		# would re-point the arm every frame and rebase the tracking with it.
		if _carrying and not _lifted and _lift_progress() >= 1.0:
			_lifted = true
		# Il waypoint scorre a ogni passo: la sua altezza e' gia' quella piena, e la sua posizione a
		# terra cammina da dove il cubo e' stato preso verso la zona in proporzione a quanta salita e'
		# stata guadagnata. Salire vale sempre qualcosa, e compra avvicinamento alla zona invece di
		# sbloccarlo tutto in una volta.
		if _carrying and not _delivered:
			_place_carry_waypoint()
		if _should_release():
			_release_object()
		else:
			_hold_captured_object()
		return
	if _carrying:
		# Let go while carrying: delivered if it came to rest inside the zone, otherwise the episode
		# simply continues -- the cube is back on the floor and can be picked up again.
		_check_delivery()
		return
	if _grasp_capture_condition():
		_capture_object()


## Progress over BOTH halves of the task, fed to the reward system by the scene's ProgressProvider.
##
## The agent's own get_progress cannot express this. It is built for reach-and-hold: once something
## is grasped it returns `share + (1 - share) * hold_fraction` and stops looking at position
## entirely. With a drop phase that means carrying the cube pays nothing at all, and the best thing
## a policy can do is grab it and freeze -- holding still is what the remaining term rewards.
##
##   0.00 - 0.50   approaching the cube
##   0.50 - 0.85   carrying it towards the drop pose
##   1.00          delivered
##
## Capped at 0.85 while carrying so `ignore_stall_above_progress` (0.9) stays out of reach until the
## cube is actually delivered: otherwise hovering over the zone forever would switch the stall guard
## off, which is the camping loophole this scene has already paid for once.
func get_task_progress(_agent: Node = null) -> float:
	if not drop_enabled:
		return float(arm.call("get_progress"))
	# Delivery is 0.90, not 1.0: the last tenth is the arm getting back home. Paying the full score
	# at the release leaves the rest of the episode flat, and the arm has no reason to leave.
	if _delivered:
		# Senza il rientro nel compito, consegnare E' finire.
		if not home_return_required:
			return 1.0
		return 0.90 + 0.10 * _home_return_progress()
	var pose_progress := clampf(float(arm.call("get_pose_progress")), 0.0, 1.0)
	# The bands do NOT meet. Approach tops out at 0.45 and carrying starts at 0.55, so the capture
	# itself is a step UP of 0.10.
	#
	# They used to meet at 0.5, and that made capturing worth exactly nothing: a perfect approach
	# already scored 0.5, and the instant the cube was caught the measurement switched to the drop
	# pose, which is far away, so progress landed back on 0.5. The reward pays on the DELTA, so the
	# grasp paid zero. Measured over 800 episodes with the hand starting ON the cube: best-in-episode
	# progress pinned at exactly 0.500, zero captures.
	if _carrying:
		# The lift is paid for BEFORE the transport, and it is worth a third of the carrying band. A
		# gate on its own would only tell the policy afterwards that a whole successful-looking
		# episode did not count; this makes rising off the floor pay from the first millimetre.
		# In parallel, and the zone is the goal from the moment of capture.
		#
		# Two sequenced versions were tried and both failed the same way. A hard gate -- the zone only
		# becomes the goal once the lift is done -- left the arm 13 mm short of the threshold forever,
		# with nothing pulling it sideways; and lowering the gate only dragged the height down with it
		# (81.8 -> 72.4 -> 49.7 across three runs). A waypoint that SLID towards the zone as the lift
		# was earned was worse still: it recedes exactly when the arm rises, so `pose_tracking` -- the
		# biggest dense term -- FALLS as the arm does the thing being asked of it. Lifting was punished.
		#
		# The order is enforced where it belongs instead: `carry_low` charges every step the held cube
		# spends below the lift height, and a delivery is refused outright unless the cube was actually
		# raised. Both of those cost nothing to an arm that lifts, and neither creates a wall.
		return 0.55 + 0.30 * (0.35 * _lift_progress() + 0.65 * pose_progress)
	return 0.45 * pose_progress


## Reads where the tool ends up with the arm at its home joints, once, and leaves the arm as it was.
##
## Measured rather than hand-written: the home pose is the agent's, and a second copy of it here
## would drift the first time anyone changes it.
func _capture_home_pose() -> void:
	var robot := arm.get_node_or_null("xarm")
	var tool: Node3D = arm.tool_pose
	if robot == null or tool == null or _home_pose == null:
		return
	var was := {}
	for joint_name in robot.get_actuated_joint_names():
		was[String(joint_name)] = robot.get_joint_position(String(joint_name))
	# The link transforms are written by _apply_kinematic_pose, which normally runs on the next
	# physics step -- reading the tool straight after reset_joint_positions returns the pose the arm
	# had BEFORE, and the captured "home" landed 434 mm from where the arm actually goes home.
	robot.reset_joint_positions(arm.call("_build_home_positions"))
	robot.call("_apply_kinematic_pose")
	_sync_tool_pose_to_tips()
	_home_pose.global_transform = Transform3D(tool.global_basis, tool.global_position)
	robot.reset_joint_positions(was)
	robot.call("_apply_kinematic_pose")
	_sync_tool_pose_to_tips()
	print("[casa] posa di ritorno a %s" % str(_home_pose.global_position))


## Sends the arm straight up from where it caught the cube.
##
## Pointing it at the drop zone the instant it captures asks for the lift and the traverse at once,
## and the cheapest way to serve both is to slide along the floor -- which is exactly what it learned
## to do. This makes the order explicit: up first, across second.
func _retarget_to_lift() -> void:
	if _lift_pose == null or grasp_target == null:
		return
	_carry_from = grasp_target.global_position
	_place_carry_waypoint()
	arm.target = _lift_pose
	arm.target_pose = _lift_pose
	# The goal just jumped; the approach term measures a DIFFERENCE and would bill the jump.
	_final_approach_last = INF
	controller.rebase_agent_tracking(arm)


## Moves the carry waypoint: up first, then across as the lift is earned.
##
## The handover used to be a cliff -- the drop zone became the goal only once the lift crossed a
## threshold, and until then nothing pulled the arm sideways at all. Measured, the arm lifted to a
## median of 67 mm against an 80 mm gate and simply stayed over the spawn: it got close to the gate,
## never crossed it, and everything behind it stayed shut. Lowering the gate made it worse, because
## the height follows the gate down (81 -> 72 -> 49 across three runs).
##
## So the waypoint slides instead. Its height is the full lift from the start, and its ground position
## walks from where the cube was caught towards the drop zone in step with how much of the lift has
## actually been achieved. Rising is always worth something, and it buys progress towards the zone
## rather than unlocking it all at once.
func _place_carry_waypoint() -> void:
	if _lift_pose == null or grasp_target == null:
		return
	var lift := _lift_progress()
	var target := _carry_from
	if drop_enabled and drop_zone != null:
		var footprint := _drop_footprint()
		var centre: Vector3 = footprint[0]
		target = Vector3(centre.x, target.y, centre.z)
	# Lo spostamento laterale comincia solo DOPO che il cubo ha superato l'ostacolo.
	#
	# Prima avanzava linearmente con la salita: a meta' quota il bersaglio era gia' a meta' strada
	# verso la zona, quindi la traiettoria richiesta era una DIAGONALE. Su una scena senza ostacoli
	# costava poco; con un rullo alto 11 cm accanto alla zona, quella diagonale ci passa dentro -- ed
	# e' proprio il gesto che si e' visto imparare, il braccio che non si alza e taglia verso la zona.
	#
	# La soglia e' in frazione di salita, non in metri, cosi' segue `carry_lift_height` invece di
	# doverla inseguire a mano: a 0.165 di quota, 0.75 mette l'inizio della traversata 1.7 cm sopra la
	# cima del rullo. Salire continua a valere -- e' `carry_low` a pagarne il prezzo, non questo -- ma
	# non compra piu' avvicinamento finche' non si e' abbastanza in alto per traversare.
	var cross := clampf((lift - carry_cross_after) / maxf(1.0 - carry_cross_after, 0.001), 0.0, 1.0)
	var across := _carry_from.lerp(target, cross)
	_lift_pose.global_transform = Transform3D(
		_demanded_basis,
		Vector3(across.x, _surface_top_y + maxf(carry_lift_height, 0.0) + 0.01, across.z))


## Sends the arm back where it started and takes the object out of its task.
func _retarget_to_home() -> void:
	if _home_pose == null:
		return
	arm.target = _home_pose
	arm.target_pose = _home_pose
	# The goal just jumped; the approach term measures a DIFFERENCE and would bill the jump.
	_final_approach_last = INF
	controller.rebase_agent_tracking(arm)


## How far the arm has got back home after the release, as 0..1.
## Nello spazio dei GIUNTI, non come distanza dell'utensile.
##
## Il successo del ritorno ha significato la POSA da quando si e' visto che il punto di casa sta
## sull'asse di rotazione: qualunque angolo di base lo raggiunge, quindi la posizione dell'utensile
## non contiene informazione sulla base. Ma il PROGRESSO continuava a misurare quel punto, e la
## discordanza era esattamente il buco in cui la base cadeva -- misurato, il braccio arrivava a 16 mm
## dal punto con la base a 103 gradi da dove doveva stare, e il gradiente non gli chiedeva altro.
##
## Misurare la stessa cosa che il gate premia rende il percorso dell'esperto monotono: il suo ritorno
## e' una camminata nello spazio dei giunti, con la base che ha l'errore piu' grande e quindi si
## chiude per prima. E' la stessa manovra che fa lui, ora con un gradiente che la segue.
func _home_return_progress() -> float:
	# IL GIUNTO PEGGIORE, la stessa quantita' che il gate del successo guarda.
	#
	# La media lasciava vincere la maggioranza: con la base a +65 e gli altri quattro a +12, la media
	# vale 0,80, mentre la posa del clone -- base a +13 e gli altri a +75 -- ne vale 0,46. Cioe' il
	# reward PREFERIVA lasciare andare la base pur di sistemare quattro giunti piccoli, ed e'
	# esattamente lo scambio che SAC ha fatto: misurato, utensile a 12,6 mm da casa con la posa a
	# 64,8 gradi.
	#
	# Col peggiore, progresso e gate misurano la stessa cosa: si sale solo migliorando il giunto messo
	# peggio, e non c'e' nessuno scambio che paghi. Il gradiente su un giunto solo alla volta -- il
	# motivo per cui avevo scelto la media -- qui non e' un problema, perche' la traiettoria completa
	# non deve piu' essere scoperta: la mostrano le dimostrazioni.
	var span := maxf(home_return_pose_span, 0.0001)
	return clampf(1.0 - _home_joint_error() / span, 0.0, 1.0)


## How far the cube has been raised towards `carry_lift_height`, as 0..1.
func _lift_progress() -> float:
	if grasp_target == null:
		return 0.0
	var resting := _surface_top_y + _object_half_extents.y
	var wanted := maxf(carry_lift_height - resting, 0.0001)
	return clampf((grasp_target.global_position.y - resting) / wanted, 0.0, 1.0)


## Builds the reset-time IK once. The gripper joint is left OUT of the chain: it is in the action
## space but not on the kinematic path to the tool, so including it adds a Jacobian column that
## cannot move the tool and stalls the solve.
func _build_reverse_ik() -> void:
	if not reverse_curriculum_enabled:
		return
	var robot := arm.get_node_or_null("xarm")
	if robot == null:
		return
	var chain := PackedStringArray()
	for joint_name in arm.get_controlled_joint_names():
		if String(joint_name) != String(arm.gripper_joint_name):
			chain.append(joint_name)
	_reverse_target = Marker3D.new()
	_reverse_target.name = "ReverseRungTarget"
	add_child(_reverse_target)
	_reverse_ik = URDFIKController.new()
	_reverse_ik.name = "ReverseRungIK"
	_reverse_ik.joint_names = chain
	_reverse_ik.tcp_link_name = String(arm.tcp_link_name)
	_reverse_ik.tcp_local_offset = arm.tcp_local_offset
	# Position only. The claws come down from above and the roll about that axis is free, which is
	# the same reason orientation is out of the task; tracking it here makes the solver walk away
	# from the cube instead of onto it.
	_reverse_ik.track_orientation = false
	# Slow on purpose. The step is integrated by hand at 1/60 s, so 5 rad/s moves a joint 83 mrad per
	# iteration -- coarse enough to oscillate around the solution instead of settling, which left the
	# rung 25-32 mm out on some spawns. 1 rad/s gives 16 mrad steps; the iteration budget covers the
	# travel either way.
	_reverse_ik.max_joint_speed = 1.0
	_reverse_ik.active = false
	add_child(_reverse_ik)
	_reverse_ik.set_robot(robot)
	_reverse_ik.set_target(_reverse_target)


## Standoff above the cube this episode starts from. Near at the beginning, retreating to the full
## task, so the assist is withdrawn as the policy stops needing it.
func reverse_standoff_for_episode(_episode: int = 0) -> float:
	if not reverse_curriculum_enabled:
		return reverse_standoff_far
	return lerpf(reverse_standoff_near, reverse_standoff_far, clampf(_reverse_level, 0.0, 1.0))


## Called when an episode actually captured the cube: the assist retreats by one notch.
func note_reverse_capture() -> void:
	if not reverse_curriculum_enabled:
		return
	var span := maxf(reverse_standoff_far - reverse_standoff_near, 0.0001)
	_reverse_level = clampf(
		_reverse_level + maxf(reverse_advance_per_capture, 0.0) / span, 0.0, 1.0)


## Places the hand on this episode's rung, by handing the agent the joint offsets that put it there.
##
## Written as OFFSETS rather than by posing the robot directly: the agent re-applies
## home + _pending_reset_offsets during its own reset, which runs after this, and would otherwise
## overwrite whatever was posed here. The scene already used that channel for joint jitter.
##
## The claws are left OPEN. A seed that starts already holding the cube teaches nothing about
## closing -- the trap the OpenArm bootstrap seeds fell into, where a 4 mm tolerance handed the
## grasp over and the policy never learned to shut the hand.
func _seed_reverse_curriculum() -> Array[float]:
	var none: Array[float] = []
	if not reverse_curriculum_enabled or _reverse_ik == null or grasp_target == null:
		# No ladder at all is a home start too. Reported as one, or an evaluation run with the
		# curriculum switched off says "0 episodes started from home" about episodes that all did.
		_home_start_episode = not reverse_curriculum_enabled
		return none
	# A share of episodes start where the arm really starts. Drawn from the episode's own RNG so a
	# seed reproduces the same mix, and never while the ladder is still at the bottom -- a policy that
	# cannot yet close the claws on a cube under its hand learns nothing from a 43 cm approach.
	if _reverse_level >= 0.25 and _episode_rng.randf() < home_start_fraction:
		_home_start_episode = true
		return none
	_home_start_episode = false
	var standoff := reverse_standoff_for_episode(_training_episode)
	# NO CLIFF AT THE END. This used to return `none` at full retreat, which drops the start from a
	# 180 mm standoff straight to the home pose 428 mm away -- not the next rung, a different task.
	# Measured on v17: nine credited captures retired the assist at EPISODE 47, and from there every
	# episode started from home. The captures did not "vanish"; the task changed under them, and the
	# policy spent the next 4900 episodes on a problem it had no shaping for. The last rung is now a
	# rung, and reaching from home stays what the pose reward already trains.
	var robot := arm.get_node_or_null("xarm")
	if robot == null:
		return none
	var home: Dictionary = arm.call("_build_home_positions")
	var joint_names: PackedStringArray = arm.get_controlled_joint_names()

	# The solver steers tcp_link + tcp_local_offset, while the task's tool point rides the claw tips.
	# Handing it the stale offset makes it drive a point 78 mm away, which converges perfectly and
	# still leaves the claws nowhere near the cube.
	_reverse_ik.tcp_local_offset = _tool_offset_in_hand()

	# The solve step is integrated HERE. `solve(apply=true)` writes joint target VELOCITIES, which
	# only become motion when a physics frame integrates them -- so a synchronous loop calling it
	# leaves the arm exactly where it started, which is what the first version of this did (the
	# probe measured the hand 443 mm from every rung, i.e. still at home). Resets cannot await
	# frames, so the velocities are stepped by hand instead.
	# Solved through the ROBOT's own kinematic step rather than by writing joint positions.
	#
	# `solve(true)` sets target velocities and the robot integrates them in _step_kinematic_control,
	# which also handles mimic joints and its own clamping. Integrating by hand instead skipped all
	# of that and stalled 25-32 mm from the rung on two spawns out of three, while the same solver
	# driven a frame at a time reaches 3 mm. Resets cannot await frames, so the robot's step is
	# called directly.
	# Several starts, not one. Damped least squares descends greedily, so from the home pose it
	# stalls 28-32 mm short on some spawns while reaching 3 mm on others -- the same behaviour the
	# stall probe measured, where only a lucky starting configuration converged. Home is tried first
	# because it usually works and costs nothing; random restarts follow until one lands.
	# The rung climbs if it has to. At standoff 0 the target is the cube's centre, 15.5 mm off the
	# floor, and for some spawns that is past the arm's reach however the solve is started: measured,
	# one spawn in three stalls at 28 mm no matter how many restarts it gets. Asking for a slightly
	# higher rung there gives a usable seed instead of no seed, and it is still far easier than the
	# full task.
	var step := 1.0 / 60.0
	var best_pose := home.duplicate()
	var best_error := INF
	var best_upright := false
	var used_standoff := standoff
	for lift in [0.0, 0.01, 0.02, 0.03]:
		used_standoff = standoff + float(lift)
		_reverse_target.global_transform = Transform3D(
			_demanded_basis, grasp_target.global_position + Vector3.UP * used_standoff)
		best_error = INF
		best_upright = false
		for attempt in range(maxi(reverse_seed_attempts, 1)):
			var start := home.duplicate()
			if attempt > 0:
				for joint_name in _reverse_ik.joint_names:
					var joint_node = robot.get_joint_node(String(joint_name))
					if joint_node == null or joint_node.joint == null or joint_node.joint.limit == null:
						continue
					start[String(joint_name)] = _episode_rng.randf_range(
						float(joint_node.joint.limit.lower), float(joint_node.joint.limit.upper))
			robot.reset_joint_positions(start)
			for _iteration in range(maxi(reverse_solve_iterations, 1)):
				_reverse_ik.solve(true)
				robot.call("_step_kinematic_control", step)
			var error := _grasp_point_world().distance_to(_reverse_target.global_position)
			# A pose that reaches the rung from ABOVE beats one that reaches it just as closely with
			# the wrist rolled under. Distance still decides between two poses on the same side; this
			# only stops a sideways branch from winning on a fraction of a millimetre.
			# Prefer the branch the arm can actually CARRY from.
			#
			# The seeding solve lands on either of two mirrored configurations for the same cube, and
			# measured across six spawns the base joint comes out at +80..+97 degrees on one and
			# -77..-79 on the other. The drop zone sits at -90 degrees of azimuth, so from the second
			# it is 13 degrees away and from the first it is 182 -- and the base has 203 degrees of
			# travel in total, so the long way round is a sweep the solver will not make in one pull.
			# That is the difference between an episode that delivers and one that stalls over the
			# robot's own centre, which is exactly what it looks like on screen.
			#
			# Uprightness was a proxy for this and only half worked. The real criterion is how far the
			# base still has to turn.
			var base_now: float = robot.get_joint_position(_base_joint_name())
			var downward := -robot_hand_basis().y
			# One DIRECTION of travel, not merely a short one.
			#
			# Both branches can reach the zone, so picking the smaller sweep chose whichever, and the
			# arm turned one way on some spawns and the other way on others. What is wanted is one
			# direction for every spawn -- but WHICH direction is a fact about where the zone is, not
			# something to write down here. The first version hardcoded "the zone lies below every
			# grasp"; the areas were then moved -- spawn in front, zone to the side -- the zone came
			# out ABOVE every grasp instead, and the rule rejected every correct pose in silence.
			# So the direction is read from the layout: from home the base turns towards the zone, and
			# every seed must sweep that same way.
			var sweep := _base_sweep_to_zone(base_now)
			var wanted_way := signf(_base_sweep_to_zone(0.0))
			var upright := (
				_wrist_posture_error(robot, base_now)
					<= deg_to_rad(maxf(wrist_reference_tolerance_degrees, 1.0))
				and sweep * wanted_way >= 0.0
				and absf(sweep) <= deg_to_rad(150.0))
			var better := (
				error < best_error
				if upright == best_upright
				else upright)
			if better:
				best_error = error
				best_upright = upright
				best_pose = {}
				for joint_name in robot.get_actuated_joint_names():
					best_pose[String(joint_name)] = robot.get_joint_position(String(joint_name))
			if upright and error <= maxf(reverse_seed_tolerance, 0.0):
				break
		if best_error <= maxf(reverse_seed_tolerance, 0.0):
			break
	robot.reset_joint_positions(best_pose)

	# Did the solve actually get there? For some spawns it does not, and a seed that leaves the claws
	# 84 mm behind the cube and 44 mm to the side is worse than no seed: the episode starts in a pose
	# the curriculum never intended, and the capture can never fire from it. Measured across three
	# spawns, one failed this way while the other two landed within 3 mm.
	var reached := best_error
	if log_capture_rejections:
		print("[piolo] risolto a %.1f mm dal piolo (limite %.1f)" % [
			reached * 1000.0, maxf(reverse_seed_tolerance, 0.0) * 1000.0])
	if reached > maxf(reverse_seed_tolerance, 0.0):
		_reverse_seed_failures += 1
		if log_capture_rejections:
			print("[piolo] seme scartato: %.1f mm dal piolo piu' basso raggiungibile (limite %.1f) -- si parte da casa" % [
				reached * 1000.0, maxf(reverse_seed_tolerance, 0.0) * 1000.0])
		robot.reset_joint_positions(home)
		return none

	var offsets: Array[float] = []
	for joint_name in joint_names:
		var name := String(joint_name)
		if name == String(arm.gripper_joint_name):
			offsets.append(0.0)  # claws stay where home leaves them: open
			continue
		offsets.append(
			robot.get_joint_position(name) - float(home.get(name, 0.0)))
	robot.reset_joint_positions(home)
	return offsets


## The grasp point in world space, computed from the CLAW LINKS rather than read off `tool_pose`.
##
## `tool_pose` is only moved in _physics_process, so inside a reset -- where no frame runs between
## posing the arm and checking it -- it still holds last frame's world position. Reading it there
## reported every rung as 445-488 mm away and threw all of them out.
func _grasp_point_world() -> Vector3:
	if _finger_tips.size() < 2:
		return (arm.tool_pose as Node3D).global_position
	var midpoint := (_tip_world(0) + _tip_world(1)) * 0.5
	return midpoint - robot_hand_basis() * maxf(tool_tip_inset, 0.0)


## The live tool point in the TCP link's frame.
func _tool_offset_in_hand() -> Vector3:
	var robot := arm.get_node_or_null("xarm")
	if robot == null:
		return arm.tcp_local_offset
	var hand := robot.get_link_node(String(arm.tcp_link_name)) as Node3D
	if hand == null:
		return arm.tcp_local_offset
	# The frozen one, so the seeding IK aims at the same point the reward measures. Reading the live
	# tip midpoint here would hand the solver a target that depends on how open the claws happen to
	# be at reset.
	if _tool_offset_frozen_valid:
		return _tool_offset_frozen
	return hand.global_transform.affine_inverse() * _grasp_point_world()


## Has the released cube settled inside the drop zone?
##
## Checked for a few frames after the release rather than on the frame it happens: the cube leaves
## the claws in mid air, and where it is at that instant is not where it ends up.
## One line per finished episode into XARM_EPISODE_LOG, if that names a file.
func _log_finished_episode() -> void:
	var path := OS.get_environment("XARM_EPISODE_LOG")
	if path.is_empty() or _capture_attempts <= 0:
		return
	var footprint := _drop_footprint()
	var centre: Vector3 = footprint[0]
	var landed := grasp_target.global_position - centre
	var home_away := 0.0
	if _home_pose != null and arm.tool_pose != null:
		home_away = (arm.tool_pose as Node3D).global_position.distance_to(
			_home_pose.global_position)
	var record := {
		"spawn_x": _spawn_position.x,
		"spawn_z": _spawn_position.z,
		"da_casa": _home_start_episode,
		"livello": _reverse_level,
		"distacco_mm": reverse_standoff_for_episode(_training_episode) * 1000.0,
		"presa": _credited_capture,
		"alzata_mm": (_carry_highest * 1000.0) if _carry_highest > -INF else 0.0,
		"consegnato": _delivered,
		"scarto_mm": [landed.x * 1000.0, landed.z * 1000.0],
		"attraversamento_mm": [_crossing_offset.x * 1000.0, _crossing_offset.y * 1000.0],
		"casa_mm": home_away * 1000.0,
		"casa_migliore_mm": (_home_closest * 1000.0) if _home_closest < INF else -1.0,
		"casa_scarto_mm": _home_offset_at_closest,
		"casa_giunti_gradi": _home_joint_errors,
		"posa_migliore_gradi": (rad_to_deg(_home_pose_best) if _home_pose_best < INF else -1.0),
		"posa_casa_gradi": rad_to_deg(_home_joint_error()),
		"presa_across_mm": _capture_across,
		"presa_varco_mm": _capture_gap,
		"presa_profondita_mm": _capture_depth,
		"presa_laterale_mm": _capture_side,
		"presa_fuori_mm": _capture_outside,
		"rilascio_punte_gradi": _release_tip_tilt,
		"rilascio_asse_gradi": _release_axis_pitch,
		"velocita_a_casa": _home_speed_at_closest,
		"gate_mm": arm.success_distance * 1000.0,
		"gate_scattato_a_mm": _gate_success_distance * 1000.0,
		"gate_bersaglio": _gate_success_target,
		# Il successo VERO dell'agente, non una mia definizione. Questo campo diceva "consegnato e
		# utensile entro il gate di distanza", che ignora sia la posa sia l'immobilita': misurato,
		# stampava `successo true` con la posa a 64,8 gradi da casa, mentre la valutazione congelata
		# leggeva 0,00%. Un campo diagnostico che puo' dire il contrario della metrica di allenamento
		# e' peggio che inutile.
		"successo": bool(arm.call("has_succeeded")),
		"utensile_entro_gate": _delivered and _home_closest <= arm.success_distance,
	}
	if not path.begins_with("/") and not path.begins_with("res://") and not path.begins_with("user://"):
		# A relative path here resolves against whatever directory the engine happens to have been
		# started in, which for a training env is not the project. Say so instead of writing nothing:
		# an empty log looks exactly like a policy that never captured anything.
		if not _episode_log_warned:
			_episode_log_warned = true
			push_warning("XARM_EPISODE_LOG deve essere un percorso assoluto: " + path)
		return
	var file := FileAccess.open(path, FileAccess.READ_WRITE)
	if file == null:
		file = FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		if not _episode_log_warned:
			_episode_log_warned = true
			push_warning("XARM_EPISODE_LOG non scrivibile: " + path
				+ " (errore " + str(FileAccess.get_open_error()) + ")")
		return
	file.seek_end()
	file.store_line(JSON.stringify(record))
	file.close()


## Charges the -5 for a cube that was being carried and did not end up in the zone.
##
## Called from the two failing branches of the delivery test and nowhere else, so it fires once per
## loss, on evidence -- the cube at rest, outside the footprint or never lifted -- rather than on the
## claws opening.
func _charge_cube_lost() -> void:
	if lost_event != null:
		lost_event.trigger(str(arm.name))


## Mette la scena nello stato che segue una consegna riuscita, senza farla.
##
## Il cubo va al centro della zona, appoggiato; lo stato interno dice consegnato; il bersaglio diventa
## casa. Il braccio parte da dove si trova a fine trasporto -- sopra la zona -- e non da casa, o non
## ci sarebbe niente da percorrere.
func _seed_delivered_state() -> Array[float]:
	var none: Array[float] = []
	if drop_zone == null or grasp_target == null or _reverse_ik == null:
		return none
	var footprint := _drop_footprint()
	var centre: Vector3 = footprint[0]
	var half: Vector3 = footprint[1]
	var resting := Vector3(centre.x, centre.y + half.y + _object_half_extents.y, centre.z)
	grasp_target.global_position = resting
	grasp_target.linear_velocity = Vector3.ZERO
	grasp_target.angular_velocity = Vector3.ZERO
	_cube_spawn = resting
	_cube_last_seen = resting
	_delivered = true
	_carrying = false
	# `_lifted` NO: il suo unico effetto qui sarebbe togliere l'esenzione alle chele, che a consegna
	# fatta non ha piu' niente da sorvegliare -- e il braccio parte proprio sopra la zona, dove un
	# tocco chiuderebbe l'episodio prima di cominciare.
	_carry_highest = carry_lift_height
	_goal_paid = true
	_retarget_to_home()
	# Il braccio sopra la zona, con le chele aperte: e' esattamente la posa in cui un rilascio lo
	# lascia, quindi il tratto da percorrere e' quello vero.
	# Dalla posa MISURATA del rilascio, non da un solve. L'IK trovava una configurazione che porta
	# l'utensile nel punto giusto, che non e' la stessa cosa: diciassette gradi di scarto sul terzo
	# giunto e la pinza in tutt'altra posizione.
	var offsets: Array[float] = []
	var index := 0
	for _joint_name in arm.get_controlled_joint_names():
		var value := (float(release_pose_degrees[index])
			if index < release_pose_degrees.size() else 0.0)
		offsets.append(deg_to_rad(value + _episode_rng.randf_range(
			-release_pose_jitter_degrees, release_pose_jitter_degrees)))
		index += 1
	return offsets


## Gli offset rispetto a casa, che e' il canale con cui l'agente posa il braccio al reset.
func _offsets_from_home(robot) -> Array[float]:
	var offsets: Array[float] = []
	var home_positions: Dictionary = arm.call("_build_home_positions")
	for joint_name in arm.get_controlled_joint_names():
		offsets.append(robot.get_joint_position(String(joint_name))
			- float(home_positions.get(String(joint_name), 0.0)))
	return offsets


func _check_delivery() -> void:
	if _delivered or drop_zone == null:
		return
	# La zona e' un VARCO, non un piano d'appoggio: sotto non c'e' niente in simulazione e nella
	# realta' c'e' il nastro trasportatore, che porta via il cubo. Il compito e' finito quando il cubo
	# ATTRAVERSA l'area, non quando si ferma dentro la sua impronta.
	#
	# La regola precedente aspettava che il cubo si fermasse e poi ne controllava x e z. Su questa
	# scena funzionava per caso -- il cubo cadeva dal varco, atterrava sul tavolo sottostante e li' si
	# fermava dentro l'impronta -- ma e' un accidente della simulazione: con un nastro sotto, quel
	# momento non arriva mai e nessuna consegna verrebbe mai contata.
	if not _zone_crossed:
		if drop_zone.overlaps_body(grasp_target):
			_zone_crossed = true
			_note_crossing_offset()
		else:
			# Caduto FUORI dal varco: il cubo si e' fermato senza esserci mai passato.
			if grasp_target.linear_velocity.length() <= maxf(drop_settle_speed, 0.0):
				_charge_cube_lost()
				_carrying = false
				_retarget_to_cube()
			return
	# Dragged, not carried. The footprint test alone has no vertical bound, so a cube pushed across
	# the floor into the zone passed it exactly like one lowered in.
	if _carry_highest < carry_lift_height:
		_charge_cube_lost()
		_carrying = false
		_retarget_to_cube()
		return
	_delivered = true
	_carrying = false
	print("[xarm_grasp_zone] cubo depositato nella zona a (%+.3f, %+.3f)" % [
		grasp_target.global_position.x, grasp_target.global_position.z])
	# The delivery is paid here and named here. `target_reached` is left for the arm getting home.
	if delivered_event != null:
		delivered_event.trigger(str(arm.name))
	else:
		goal_event.trigger(str(arm.name))
	# Senza il rientro nel compito il bersaglio NON si sposta a casa: il gate di posizione dell'agente
	# lo seguirebbe, e il successo tornerebbe a pretendere il viaggio che si e' deciso di non far
	# imparare. Lasciandolo sulla posa di rilascio, il braccio e' gia' dove il gate lo vuole e
	# l'episodio si chiude sulla consegna.
	if home_return_required:
		_retarget_to_home()
	elif _drop_pose != null and arm.tool_pose != null:
		# Senza rientro, il compito finisce QUI: il bersaglio diventa il punto in cui il braccio si
		# trova nell'istante del rilascio, cosi' il gate di posizione e' soddisfatto dove sta e resta
		# da soddisfare solo l'immobilita'. Lasciarlo sulla superficie della zona chiedeva invece di
		# scendere di altri 80 mm dopo aver lasciato il cubo -- cosa che le dimostrazioni non fanno e
		# che non ha alcuno scopo -- e infatti il successo non scattava mai: misurato, progresso 1.000
		# e selection_success 0,00%.
		_drop_pose.global_transform = (arm.tool_pose as Node3D).global_transform
		arm.target = _drop_pose
		arm.target_pose = _drop_pose
		_final_approach_last = INF
		controller.rebase_agent_tracking(arm)


## How far off centre the cube was when it crossed the zone, and what that is worth.
##
## Called once, on the frame `overlaps_body` first says the cube is in the gap.
##
## Measured AT THE CROSSING on purpose. The release point ignores the throw -- a cube let go over
## the centre while the arm is still swinging crosses the plane well off it -- and where the cube
## finally rests is after the fall, bounces included: on v61 that number put a delivery at 127 mm
## from a centre it had passed cleanly.
func _note_crossing_offset() -> void:
	var footprint := _drop_footprint()
	var centre: Vector3 = footprint[0]
	var half: Vector3 = footprint[1]
	var offset := grasp_target.global_position - centre
	_crossing_offset = Vector2(offset.x, offset.z)
	# Normalised by the half extents, so the shape follows the zone instead of assuming it square.
	# 0 at the centre, 1 at the edge, and past 1 when only a corner of the cube clipped the gap.
	var reach := maxf(
		absf(offset.x) / maxf(half.x, 0.0001), absf(offset.z) / maxf(half.z, 0.0001))
	_crossing_reward_pending = clampf(1.0 - reach, 0.0, 1.0)


## Paid ONCE, for crossing the zone near its middle rather than clipping the edge.
##
## The delivery itself is a boolean -- `overlaps_body` -- so a policy that can just catch the rim
## has no reason to aim better, and v61 did exactly that: |z| median 45.9 mm against a 50 mm half
## extent, with the worst delivery outside the footprint altogether, caught by a corner. This puts
## a gradient where the criterion has none.
##
## ADDED to the delivery, never substituted for it, and never negative: no amount of bad aim can
## make delivering worth less than keeping the cube, which is what a penalty here would risk.
##
## One shot, like the delivery. Paid as a level it would reward hovering in the gap.
func get_drop_centring_reward() -> float:
	var value := _crossing_reward_pending
	_crossing_reward_pending = 0.0
	return value


## The hand's own approach axis in world space: the direction the claws point.
func robot_hand_basis() -> Vector3:
	var robot := arm.get_node_or_null("xarm")
	if robot == null:
		return Vector3.UP
	var hand := robot.get_link_node(String(arm.tcp_link_name)) as Node3D
	if hand == null:
		return Vector3.UP
	return hand.global_basis.orthonormalized().y.normalized()


## Centre and half extents of the drop zone's footprint, from its collision shape.
func _drop_footprint() -> Array:
	for child in drop_zone.get_children():
		var shape_node := child as CollisionShape3D
		if shape_node != null and shape_node.shape is BoxShape3D:
			return [
				shape_node.global_position,
				(shape_node.shape as BoxShape3D).size * 0.5
					* shape_node.global_basis.get_scale().abs()]
	return [drop_zone.global_position, Vector3.ONE * 0.05]


## Point the arm at the drop pose instead of the cube, and tell the controller the goal moved.
func _retarget_to_drop() -> void:
	if not drop_enabled or drop_zone == null or _drop_pose == null:
		return
	var footprint := _drop_footprint()
	var centre: Vector3 = footprint[0]
	var half: Vector3 = footprint[1]
	_drop_pose.global_transform = Transform3D(
		_demanded_basis,
		Vector3(centre.x, centre.y + half.y + maxf(drop_release_height, 0.0), centre.z))
	arm.target = drop_zone
	arm.target_pose = _drop_pose
	# The goal just jumped; the approach term measures a DIFFERENCE and would bill the jump.
	_final_approach_last = INF
	controller.rebase_agent_tracking(arm)


func _retarget_to_cube() -> void:
	arm.target = grasp_target
	arm.target_pose = grasp_target_pose
	# The goal just jumped; the approach term measures a DIFFERENCE and would bill the jump.
	_final_approach_last = INF
	controller.rebase_agent_tracking(arm)


## Geometric capture: the object inside the claws, and the claws closed onto its width.
##
## The proximity rule this replaces asked for the object at the tool, the claws past 60% shut, AND
## the object still. Those three cannot hold at once here. The claws are kinematic -- infinite mass --
## so their first contact kicks the cube, and the kick lands well before the closure fraction reaches
## the threshold: by the time the second condition is true the third is already false. Measured with
## a perfect IK controller, the descent left the cube untouched (0.0 mm) and the closure then sent it
## 22 metres, on every attempt.
##
## The geometric test fires at the moment the claws ENCLOSE the cube, which is before contact rather
## than after, so there is no impulse to survive.
##
## This was not available until the claws were measured correctly. `gripper_finger_link_names` named
## grip_left/right -- the linkage arms -- and off those the pad gap read -5 mm at every joint angle,
## a number no geometric rule can use. Off finger_left/right it runs -5.3 mm shut to 54.0 mm open,
## and a 30 mm cube fits.
##
## The OpenArm's own rule is a separate function in its own scenario; nothing here touches it.
func _grasp_capture_condition() -> bool:
	if not grasp_capture_geometric:
		return _proximity_capture_condition()
	if _finger_tips.size() < 2:
		return _proximity_capture_condition()
	_capture_attempts += 1
	# 1. The cube sits between the claws, measured against the CLAWS themselves.
	#
	# Not against the tool frame: the tool rides the claw tips, so it travels 28.9 mm as they shut,
	# and the offset from tool to a cube standing perfectly still grows from 0 to -25 mm during the
	# closure. The cube would leave the window precisely while the claws close around it. The claws
	# are the thing doing the grasping, so they are what the cube is measured against.
	var left := _tip_world(0)
	var right := _tip_world(1)
	var span := left - right
	if span.length() < 0.0001:
		return _proximity_capture_condition()
	var closing_axis := span.normalized()
	var centre := (left + right) * 0.5
	var to_cube := grasp_target.global_position - centre
	# Across the claws: room to spare is (gap - width) / 2, since both claws must reach it.
	var across := absf(to_cube.dot(closing_axis))
	var half_room := maxf(
		(float(arm.call("pad_gap")) - float(arm.grasp_object_width)) * 0.5, 0.0)
	# Il cubo dev'essere FRA le due punte, non solo vicino al loro punto di mezzo. Visto a schermo: il
	# blocco preso male e rimasto attaccato a UNA chela. La distanza dal punto di mezzo lo permette
	# quando il varco e' largo -- l'indulgenza (varco - larghezza)/2 + tolleranza cresce col varco e a
	# varco aperto supera meta' varco, cioe' la posizione di un cubo appoggiato a una sola punta.
	# Questo invece chiede il segno: il cubo sta da parti opposte rispetto alle due punte, sempre.
	# Con una tolleranza, non a segno secco. Il test a segno secco rifiutava anche il cubo che sporge
	# di un paio di millimetri oltre una punta ma che la chiusura racchiude comunque, e misurato in
	# allenamento toglieva alla policy quasi meta' delle catture di striscio -- quelle da cui il
	# critico impara che cosa sia una presa. Cinque millimetri lasciano passare quelle e continuano a
	# escludere il caso per cui la prova esiste: un cubo appoggiato FUORI da una chela sta a piu' di
	# quindici millimetri dalla punta.
	var slack := 0.005
	var side_of_left := (grasp_target.global_position - left).dot(closing_axis)
	var side_of_right := (grasp_target.global_position - right).dot(closing_axis)
	if side_of_left * side_of_right > 0.0 and minf(absf(side_of_left), absf(side_of_right)) > slack:
		# Contato a parte dalla prova della distanza: sono due domande diverse -- "fra le punte" e
		# "abbastanza al centro" -- e un contatore solo non dice quale delle due sta rifiutando.
		_reject_between += 1
		_best_between = minf(_best_between, minf(absf(side_of_left), absf(side_of_right)))
		return false
	if across > half_room + maxf(grasp_capture_tolerance, 0.0):
		_reject_across += 1
		_best_across = minf(_best_across, across - half_room)
		return false
	# Along the claws: the cube must lie in the volume they enclose, which is NOT a fixed distance
	# from their tips. Closing advances the tips 29 mm while the cube stays put, so a straight
	# distance test tightens exactly as the grasp completes: measured with the claws shut on a
	# perfectly placed cube, it read 76.6 mm against a 56 mm limit and refused a good grasp.
	#
	# Split instead. Depth runs along the hand axis: the cube may sit well BEHIND the tips (that is
	# where a held object ends up) but not beyond them. The remaining component is the sideways miss.
	# Depth and sideways are measured from the FROZEN tool point, not from the live tip midpoint.
	#
	# The tips travel on an arc as the claws shut, so the midpoint moves forward AND swings off the
	# hand axis; measured with the claws fully closed on a cube standing still, the sideways component
	# read 37-45 mm against a 30 mm limit, and 800 of 900 capture attempts an episode were refused on
	# it. The question these two tests ask -- is the cube inside the volume the claws enclose -- has
	# nothing to do with how far through the closure we are, so it must not be asked from a point that
	# moves with it. Fourth instance of that trap in this file.
	#
	# The ACROSS test above keeps the live tips on purpose: its allowance is (gap - width) / 2, so
	# both sides of that comparison move together and the pairing stays honest.
	var anchor: Vector3 = (arm.tool_pose as Node3D).global_position
	var to_anchor := grasp_target.global_position - anchor
	var hand := robot_hand_basis()
	# DEPTH follows the LIVE tips; SIDEWAYS stays on the frozen point. They are different questions.
	#
	# "Is the cube between the pads, right now" has to move with the pads: the tips advance ~29 mm as
	# the claws shut, and measured from the frozen point -- which sits where the tips are when OPEN --
	# a capture was being declared with the cube 20 to 29 mm BEHIND the closed tips, up at the knuckle.
	# That is what the grasp looked like on screen.
	#
	# "Is the cube off to one side" must NOT move with them: the tip midpoint travels on an arc, so it
	# swings off the hand axis as well as forward, and read from there the sideways miss showed 37-45
	# mm on a cube standing still and refused 800 of 900 attempts an episode.
	var depth := (grasp_target.global_position - centre).dot(hand)
	if depth > maxf(grasp_capture_ahead, 0.0) or depth < -maxf(grasp_capture_depth, 0.0):
		# Counted apart from the sideways miss below. They were one counter and one "best", and a
		# 30 mm reading was then unattributable: too far in FRONT of the tips and too far to the SIDE
		# have different causes and different fixes, and the merged number said neither.
		_reject_depth += 1
		_best_depth = minf(_best_depth, absf(depth))
		return false
	var sideways := (to_anchor - hand * depth - closing_axis * to_anchor.dot(closing_axis)).length()
	if sideways > maxf(grasp_capture_distance, 0.0):
		_reject_side += 1
		_best_side = minf(_best_side, sideways)
		return false
	# 2. The claws have closed ONTO it: a crossing, not a window. The gap shrinks several mm per
	# physics frame, so asking for |gap - width| inside a band would make capture depend on the
	# frame rate. Closing further does not cancel the grasp -- a kinematic claw cannot be stopped by
	# contact, so it always ends up inside the object.
	var fit_error: float = float(arm.call("pad_gap")) - float(arm.grasp_object_width)
	if fit_error > maxf(grasp_capture_gap_tolerance, 0.0):
		_reject_gap += 1
		_best_gap = minf(_best_gap, fit_error)
		return false
	if fit_error < -maxf(grasp_capture_gap_overclose, 0.0):
		# Shut PAST the cube rather than onto it.
		_reject_gap += 1
		_best_gap = minf(_best_gap, absf(fit_error))
		return false
	# 3. Still at rest. Kept, but it is no longer the impossible condition it was under the proximity
	# rule: this fires before the claws touch, so the cube has not been kicked yet.
	if grasp_target.linear_velocity.length() > maxf(grasp_capture_max_object_speed, 0.0):
		_reject_speed += 1
		_best_speed = minf(_best_speed, grasp_target.linear_velocity.length())
		return false
	if log_capture_rejections:
		# The same instant seen from the two frames that matter. `depth` is measured from the FROZEN
		# tool point, which sits where the tips are when the claws are OPEN; the tips themselves
		# advance ~29 mm as they shut. If the cube is well in front of the live tips at capture, the
		# grasp is being declared before the pads have reached it.
		var frozen := grasp_target.global_position - (arm.tool_pose as Node3D).global_position
		print("[presa] catturato | profondita dalle punte VIVE %+.4f m (dal punto congelato %+.4f) | varco-larghezza %+.4f m" % [
			depth, frozen.dot(hand), fit_error])
	return true


## The old rule, kept behind the flag so the change is one switch to undo.
func _proximity_capture_condition() -> bool:
	var tool: Node3D = arm.tool_pose
	if tool == null:
		return false
	if tool.global_position.distance_to(grasp_target.global_position) > maxf(
			grasp_capture_distance, 0.0):
		return false
	if _gripper_closure() < clampf(grasp_capture_closure, 0.0, 1.0):
		return false
	if grasp_target.linear_velocity.length() > maxf(grasp_capture_max_object_speed, 0.0):
		return false
	return true


## Fraction of the gripper's travel towards shut: 0 fully open, 1 fully closed.
func _gripper_closure() -> float:
	if not arm.has_method("grasp_clearance_normalised"):
		return 0.0
	var travel: float = float(arm.call("_gripper_position"))
	var limit: float = float(arm.call("_gripper_travel_limit"))
	if limit <= 0.0:
		return 0.0
	return clampf(1.0 - travel / limit, 0.0, 1.0)


func _should_release() -> bool:
	# Hysteresis, so a grasp sitting on the threshold does not chatter between held and dropped.
	if not grasp_capture_geometric or arm.call("grasp_window").is_empty():
		return _gripper_closure() < clampf(
			grasp_capture_closure - maxf(grasp_release_clearance, 0.0), 0.0, 1.0)
	# The inverse of capture condition 2: the claws have opened clear of the object again.
	var fit_error: float = float(arm.call("pad_gap")) - float(arm.grasp_object_width)
	return fit_error > maxf(grasp_capture_gap_tolerance, 0.0) + maxf(
		grasp_release_gap_clearance, 0.0)


func _capture_object() -> void:
	var tool: Node3D = arm.tool_pose
	if tool == null:
		return
	_record_capture_geometry()
	_grasp_attach_transform = tool.global_transform.affine_inverse() * grasp_target.global_transform
	_grasp_frozen_before = grasp_target.freeze
	_grasp_gripper_held = false
	if grasp_attach_kinematic:
		grasp_target.freeze_mode = RigidBody3D.FREEZE_MODE_KINEMATIC
		grasp_target.freeze = true
	arm.set_grasp_attached(true)
	# The claws stop where they caught it.
	#
	# A control step is three physics frames and the gripper runs at 8 rad/s, so the capture fires at
	# a pad gap of about 29 mm on a 30 mm cube and the SAME step carries the claws on to 16 mm -- 14 mm
	# inside the object. The rule was satisfied at the right moment and the picture was wrong a
	# fraction of a second later, which is what kept looking like over-squeezing.
	var robot := arm.get_node_or_null("xarm")
	if robot != null:
		_grasp_gripper_position = robot.get_joint_position(String(arm.gripper_joint_name))
		_grasp_gripper_held = true
	_hold_captured_object()
	if not _credited_capture:
		_credited_capture = true
		note_reverse_capture()
		if grasp_event != null:
			grasp_event.trigger(str(arm.name))
	if drop_enabled and not _delivered:
		_carrying = true
		# SU PRIMA, POI DI TRAVERSO -- e finora non succedeva. `_retarget_to_lift` e il waypoint che
		# scorre erano scritti, documentati e MAI CHIAMATI: la cattura puntava dritto alla zona di
		# rilascio, quindi il modo piu' economico di servire sia la salita sia la traversata era
		# strisciare sul pavimento. Misurato in v52: 122 prese su 200 episodi, ZERO consegne, e 34
		# sfondamenti del pavimento col cubo in mano su 171 episodi.
		_retarget_to_lift()


func _release_object() -> void:
	_record_release_attitude()
	# La posa dei giunti NELL'ISTANTE del rilascio: e' da li' che il ritorno comincia davvero, ed e'
	# il metro con cui giudicare la posa seminata dalla scena del solo ritorno. Se le due non
	# coincidono, quel tratto viene allenato da un punto di partenza che il compito intero non
	# produce.
	var robot_rel := arm.get_node_or_null("xarm")
	if robot_rel != null:
		var home_rel: Dictionary = arm.call("_build_home_positions")
		var pose_rel := []
		for joint_name in arm.get_controlled_joint_names():
			pose_rel.append("%+.1f" % rad_to_deg(
				robot_rel.get_joint_position(String(joint_name))
				- float(home_rel.get(String(joint_name), 0.0))))
		print("[rilascio] posa (base per prima): %s" % " ".join(PackedStringArray(pose_rel)))
	_grasp_gripper_held = false
	grasp_target.freeze = _grasp_frozen_before
	arm.set_grasp_attached(false)
	# The charge for losing the cube is NOT made here. At the instant the claws open the cube has not
	# landed yet, so `_delivered` is still false whatever the release was -- and the correct release
	# over the zone was being billed exactly like a cube dropped on the floor. It is charged in
	# `_check_delivery` instead, where the cube has come to rest and it is known which of the two it
	# was. (Measured before moving it: releasing still paid +49.3 more than holding on, so this was
	# never what kept the arm from letting go -- but a cost on the right behaviour points the wrong
	# way and had to go.)
	# Let it fall like an object once it is out of the hand.
	#
	# The three angular locks exist for the APPROACH: a cube that yaws while being nudged presents its
	# 42 mm diagonal instead of its 30 mm face and the claws stop fitting round it. Once released none
	# of that matters, and keeping them on made the drop look wrong -- the cube snapped to axis-aligned
	# and descended perfectly upright, which is not what a dropped block does.
	if cube_free_after_release:
		grasp_target.axis_lock_angular_x = false
		grasp_target.axis_lock_angular_y = false
		grasp_target.axis_lock_angular_z = false


## Replays the captured relative pose: the same mechanism the arm already uses for its own links.
func _hold_captured_object() -> void:
	# Hold the grip as well as the pose: without this the claws keep travelling on whatever command
	# the policy is still sending, and a held cube is squeezed further with every step.
	var robot := arm.get_node_or_null("xarm")
	# Il fermo contro la chiusura esiste perche' con la presa DICHIARATA stringere di piu' non serve a
	# niente e schiaccia il cubo. Con la presa fisica e' il contrario: a tenerlo e' la stretta, quindi
	# il fermo va tolto o il cubo scivola via appena il braccio accelera.
	if _grasp_gripper_held and robot != null and grasp_attach_kinematic:
		# A stop against CLOSING FURTHER, not a lock.
		#
		# Holding the joint at the captured value flatly meant the claws could never open again: the
		# release leg commanded +1.0 for eighteen frames and nothing happened, so the cube was carried
		# to the zone, taken home and still in the hand. Opening is always allowed -- that is how a
		# grasp ends -- and only travel past where it caught is refused.
		var gripper := String(arm.gripper_joint_name)
		var now: float = robot.get_joint_position(gripper)
		var closes_negative := _grasp_gripper_position < 0.0
		var deeper := now < _grasp_gripper_position if closes_negative else now > _grasp_gripper_position
		if deeper:
			# Only the gripper: `reset_joint_positions` writes EVERY actuated joint and defaults the
			# ones it is not given to zero, which folded the whole arm the first time this was tried.
			var pose := {}
			for joint_name in robot.get_actuated_joint_names():
				pose[String(joint_name)] = robot.get_joint_position(String(joint_name))
			pose[gripper] = _grasp_gripper_position
			robot.reset_joint_positions(pose)
	# Presa fisica: da qui in poi il cubo lo tengono l'attrito e la stretta, non un incollaggio.
	if not grasp_attach_kinematic:
		return
	var tool: Node3D = arm.tool_pose
	if tool == null:
		return
	var pose := tool.global_transform * _grasp_attach_transform
	var rid := grasp_target.get_rid()
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_TRANSFORM, pose)
	PhysicsServer3D.body_set_state(
		rid, PhysicsServer3D.BODY_STATE_LINEAR_VELOCITY, Vector3.ZERO)
	PhysicsServer3D.body_set_state(
		rid, PhysicsServer3D.BODY_STATE_ANGULAR_VELOCITY, Vector3.ZERO)
	grasp_target.global_transform = pose


## Per-step cost in [-1, 0] for MOVING the object, charged before and during the grasp, zero once
## held -- after a capture the arm is supposed to move it.
##
## It charges the cube's motion THIS STEP, not how far it has ended up from its spawn. The standing
## version was tried and it backfired: measured over v14, the cube stopped being shoved (median
## displacement 0.0 mm, only 2% of episodes past 20 mm, against 38% before) and the hand walked away
## from it -- median pose error 78 mm at episode 400, 146 mm by 2600, 175 mm by 5000, with captures
## falling 11% -> 1%. A charge that keeps billing for a bump made at step 5 is cheapest to avoid by
## never approaching, and the policy found that first.
##
## Per-step motion has no such fixed point: a gentle approach is free however close it gets, a shove
## costs once and stops costing when the cube stops moving.
func get_grasp_disturbance_penalty() -> float:
	if grasp_target == null or arm.is_grasp_attached() or not _demanded_basis_valid:
		return 0.0
	var here := grasp_target.global_position
	if _delivered:
		# After the delivery the piece is finished business: the arm is on its way home and must not
		# touch it again. Charged on the same per-step motion, on its own tighter scale.
		var moved_after := here - _cube_last_seen
		_cube_last_seen = here
		moved_after.y = 0.0
		return -clampf(
			moved_after.length() / maxf(delivered_disturbance_scale, 0.0001), 0.0, 1.0)
	var shift := here - _cube_last_seen
	_cube_last_seen = here
	shift.y = 0.0
	return -clampf(shift.length() / maxf(grasp_disturbance_scale, 0.0001), 0.0, 1.0)


## Per-step reward in [0, 1] for the last few centimetres onto the cube.
##
## Deliberately steep and short: it pays nothing until the hand is genuinely close, then rises
## linearly to 1 at the cube. It exists to put a gradient where the pose shaping has none, and it
## stops the moment the cube is caught -- after that the task is the lift, not the approach.
func get_final_approach_reward() -> float:
	# Whatever the CURRENT goal is, not just the cube.
	#
	# It was scoped to the approach, and that left the rest of the task with only the pose term, whose
	# gradient is 1/near_target_distance -- shallow by design so it can span the traverse. Measured on
	# v35: the arm grasps in 94% of episodes and then lifts to a median of 49.7 mm against a 70 mm
	# gate, 4% clearing it. Closing the last 50 mm onto the lift pose was worth 0.0375 of reward, so
	# it did not bother, and the traverse behind the gate never opened. Lowering the gate only chased
	# the height down: 81.8 -> 72.4 -> 49.7 across three runs.
	var target := arm.target_pose as Node3D
	var tool: Node3D = arm.tool_pose
	if target == null or tool == null:
		_final_approach_last = INF
		return 0.0
	var span := maxf(final_approach_span, 0.0001)
	var away := tool.global_position.distance_to(target.global_position)
	if away > span:
		_final_approach_last = INF
		return 0.0
	var previous := _final_approach_last
	_final_approach_last = away
	if previous == INF:
		return 0.0
	# The CLOSING of the gap, not the being close. Paid as a level it rewarded parking on the cube:
	# measured, standing still scored 40.91 against 25.04 for actually capturing, because the term
	# kept paying for 300 steps and a capture switched it off. As a difference it pays once for the
	# approach, charges the same for backing away, and is worth nothing at all to a policy that sits
	# there -- which is what shaping is supposed to be.
	return clampf((previous - away) / span, -1.0, 1.0)


## The joint the whole arm turns on: the root of the URDF chain, not a name written down.
func _base_joint_name() -> String:
	var names: PackedStringArray = arm.get_controlled_joint_names()
	return String(names[0]) if names.size() > 0 else ""


## Il ROLL della mano attorno al proprio asse di avvicinamento, in radianti.
##
## Chiesto: il pitch va bene, il roll no. La versione precedente confrontava DUE assi della mano
## contro un riferimento fisso, che e' una misura di orientamento intero: mescolava il roll con il
## pitch, quindi una posa che semplicemente allungava di piu' leggeva "postura sbagliata" e una
## davvero girata poteva passare. Peggio, il riferimento era stato ricavato dal grappolo delle pose
## che l'IK produceva -- misurato dopo: sta a 180 gradi dalla posa di CASA su entrambi gli assi --
## quindi era d'accordo con la mano storta che si vede a schermo.
##
## Qui il pitch e' libero per costruzione: si proietta tutto sul piano perpendicolare all'asse di
## avvicinamento, dove il pitch non esiste, e si guarda solo dove punta la linea fra le due dita.
func _wrist_posture_error(robot, base_angle: float) -> float:
	var hand := robot.get_link_node(String(arm.tcp_link_name)) as Node3D
	if hand == null:
		return PI
	var local := (Basis(Vector3.UP, base_angle).inverse() * hand.global_basis).orthonormalized()
	# Misurati sul modello: le chele puntano lungo -Y della mano, le dita si aprono lungo +Z.
	var approach := -local.y
	var fingers := local.z
	var wanted := wrist_reference_fingers.normalized()
	wanted = wanted - approach * wanted.dot(approach)
	var have := fingers - approach * fingers.dot(approach)
	if wanted.length() < 0.0001 or have.length() < 0.0001:
		return 0.0
	var angle := have.normalized().angle_to(wanted.normalized())
	# Una pinza girata di mezzo giro attorno al proprio asse fa la stessa presa: le dita si scambiano.
	return minf(angle, PI - angle)


## La geometria della presa NELL'ISTANTE in cui scatta.
##
## Chiesto: guardare quando il braccio prende male. Le regole di cattura hanno quattro condizioni e i
## contatori dicono solo quante volte hanno rifiutato; quando una presa storta viene ACCETTATA non
## resta niente da leggere. Questi sono i numeri di quel momento, uno per episodio.
func _record_capture_geometry() -> void:
	if _finger_tips.size() < 2:
		return
	var left := _tip_world(0)
	var right := _tip_world(1)
	var span := left - right
	if span.length() < 0.0001:
		return
	var axis := span.normalized()
	var centre := (left + right) * 0.5
	var hand := robot_hand_basis()
	var to_cube := grasp_target.global_position - centre
	_capture_across = absf(to_cube.dot(axis)) * 1000.0
	_capture_gap = float(arm.call("pad_gap")) * 1000.0
	_capture_depth = to_cube.dot(hand) * 1000.0
	var anchor: Vector3 = (arm.tool_pose as Node3D).global_position
	var to_anchor := grasp_target.global_position - anchor
	var depth := to_cube.dot(hand)
	_capture_side = (to_anchor - hand * depth - axis * to_anchor.dot(axis)).length() * 1000.0
	# Quanto il cubo sporge oltre la punta piu' vicina: positivo vuol dire che non e' fra le due.
	_capture_outside = maxf(
		(grasp_target.global_position - left).dot(axis),
		-(grasp_target.global_position - right).dot(axis)) * 1000.0


## Come sta la mano nell'istante in cui lascia il cubo.
##
## Chiesto: sopra la zona le chele devono essere parallele al pavimento. Ci sono due letture -- la
## linea fra le punte orizzontale, o l'asse di presa coricato -- e sono lavori diversi, quindi qui si
## registrano tutte e due su ogni episodio invece di sceglierne una a indovinare.
func _record_release_attitude() -> void:
	if _finger_tips.size() < 2:
		return
	var between := _tip_world(1) - _tip_world(0)
	if between.length() < 0.0001:
		return
	_release_tip_tilt = rad_to_deg(asin(clampf(absf(between.normalized().y), 0.0, 1.0)))
	var hand := arm.get_node_or_null("xarm")
	if hand == null:
		return
	var tcp := hand.get_link_node(String(arm.tcp_link_name)) as Node3D
	if tcp != null:
		_release_axis_pitch = rad_to_deg((-tcp.global_basis.y).angle_to(Vector3.DOWN))


## Radians the base still has to turn to face the drop zone, from a given angle.
func _base_sweep_to_zone(base_now: float) -> float:
	if drop_zone == null:
		return 0.0
	var footprint := _drop_footprint()
	var zone: Vector3 = footprint[0]
	# MEASURED mapping, not assumed: sweeping the base with the arm extended gives
	# world_azimuth = 90 degrees - base_angle, exactly, over the whole range
	# (base -90 -> azimuth +180, base 0 -> +90, base +90 -> 0).
	#
	# The first version of this compared the base angle against the azimuth directly, as if they were
	# the same quantity. They are not, and the error pushed seeds hard against the -100 degree limit,
	# where the grasp itself then fails.
	var wanted := PI * 0.5 - atan2(zone.z, zone.x)
	var sweep := wanted - base_now
	while sweep > PI:
		sweep -= TAU
	while sweep < -PI:
		sweep += TAU
	return sweep


## How far the claw tips travel between open and shut, in metres.
##
## Measured from the geometry rather than written down: it is the gap between the frozen tool point,
## which sits where the tips are when OPEN, and where the tips actually are now.
func _closing_sweep() -> float:
	if _finger_tips.size() < 2 or arm.tool_pose == null:
		return 0.0
	var tips := (_tip_world(0) + _tip_world(1)) * 0.5
	return absf((tips - (arm.tool_pose as Node3D).global_position).dot(robot_hand_basis()))


## Per-step reward in [0, 1] for holding the wrist the right way up.
##
## Ora legge lo stesso roll che filtra le semine, invece di una componente dell'asse X della mano
## contro l'alto del mondo: quella cambiava anche solo cambiando il pitch, quindi premiava e puniva
## cose che non erano il roll.
func get_wrist_upright_reward() -> float:
	var robot := arm.get_node_or_null("xarm")
	if robot == null:
		return 0.0
	var hand := robot.get_link_node(String(arm.tcp_link_name)) as Node3D
	if hand == null:
		return 0.0
	var roll := _wrist_posture_error(robot, robot.get_joint_position(_base_joint_name()))
	var limit := deg_to_rad(maxf(wrist_reference_tolerance_degrees, 1.0))
	return clampf(1.0 - roll / limit, 0.0, 1.0)


## Per-step cost in [-1, 0] for moving the claws while carrying.
func get_carry_gripper_still_penalty() -> float:
	if grasp_target == null or not arm.is_grasp_attached():
		return 0.0
	# Free over the drop zone. This term asks the claws to hold still while carrying, which is right
	# everywhere except the one place where opening them is the whole point -- and charged there too
	# it taught the arm to arrive with the cube and keep it: measured on v41, 95% of episodes reached
	# the release pose holding the cube and 0% let go. Over the footprint the gripper is the policy's
	# business again.
	if _above_drop_zone():
		return 0.0
	var index := Array(arm.get_controlled_joint_names()).find(String(arm.gripper_joint_name))
	var applied: Array = arm.get_previous_action_observation()
	if index < 0 or index >= applied.size():
		return 0.0
	return -clampf(absf(float(applied[index])) * maxf(carry_gripper_still_scale, 0.0), 0.0, 1.0)


## Is the held cube over the drop zone's footprint, whatever its height?
func _above_drop_zone() -> bool:
	if drop_zone == null or grasp_target == null:
		return false
	var footprint := _drop_footprint()
	var centre: Vector3 = footprint[0]
	var half: Vector3 = footprint[1]
	var offset := grasp_target.global_position - centre
	return absf(offset.x) <= half.x and absf(offset.z) <= half.z


## Per-step cost in [-1, 0] for carrying the cube too low.
##
## Only while it is actually held and not yet delivered. The release itself is free from any height:
## dropping the cube into the zone from above is a perfectly good delivery, and the settle test
## already decides whether it landed inside.
func get_carry_low_penalty() -> float:
	if grasp_target == null or not arm.is_grasp_attached() or _delivered:
		return 0.0
	# Only AFTER the lift has been achieved once. Charging it from the moment of capture taxes the
	# grasp itself: the reward probe, which catches the cube at step 2 and then holds it, scored 0.87
	# against 7.01 for standing still -- a policy that has learned to grasp but not yet to lift would
	# be punished for the half it had got right. What is being punished is LETTING IT BACK DOWN.
	if not _lifted:
		return 0.0
	var below := maxf(carry_lift_height, 0.0) - grasp_target.global_position.y
	if below <= 0.0:
		return 0.0
	return -clampf(below / maxf(carry_low_scale, 0.0001), 0.0, 1.0)


## Per-step cost in [-1, 0] for digging the claws into the ground.
##
## Charged on DEPTH BELOW THE SURFACE only: touching down at y=0 is free, which is what a grasp of a
## cube resting on the floor needs. Everything below that is cost.
## Costo per passo in [-1, 0]: quanto la posa dei giunti differisce da quella di casa, dopo la
## consegna. Zero prima, perche' durante la presa e il trasporto la posa DEVE essere diversa.
## Anche questo in premio, e solo VICINO a casa: applicato per tutta la coda a piena forza era
## l'accumulatore che rendeva la consegna un cattivo affare.
func get_home_pose_reward() -> float:
	if not _delivered or _home_pose == null or arm.tool_pose == null:
		return 0.0
	var away := (arm.tool_pose as Node3D).global_position.distance_to(_home_pose.global_position)
	# Stessa finestra stretta del premio di immobilita', e per lo stesso motivo.
	var window := maxf(success_distance * 2.0, 0.001)
	var nearness := clampf(1.0 - away / window, 0.0, 1.0)
	if nearness <= 0.0:
		return 0.0
	return nearness * (1.0 - clampf(
		_home_joint_error() / maxf(home_pose_penalty_scale, 0.0001), 0.0, 1.0))


## Errore medio, in radianti, fra la posa attuale e quella di casa. La pinza esclusa: a fine episodio
## e' aperta perche' ha appena lasciato il cubo, e chiederle di tornare come all'inizio sarebbe
## chiederle di richiudersi per niente.
func _home_joint_error() -> float:
	var robot := arm.get_node_or_null("xarm")
	if robot == null:
		return 0.0
	var home: Dictionary = arm.call("_build_home_positions")
	if home.is_empty():
		return 0.0
	# Il PEGGIORE, non la media. Misurato sul ritorno isolato: il braccio arriva a 19 mm dal punto di
	# casa con la BASE a +103 gradi da dove dovrebbe stare -- non si rigira affatto, ci arriva
	# ripiegando gli altri giunti. La media su sei giunti nascondeva un errore di cento gradi su uno.
	#
	# E il punto di casa non puo' vincolarla: sta praticamente sull'asse di rotazione (x 0,0 mm,
	# z 4,3 mm), quindi QUALUNQUE angolo di base lo raggiunge. La posizione dell'utensile non contiene
	# informazione sulla base, e finche' "casa" ha significato solo un punto, la base e' rimasta
	# libera.
	var worst := 0.0
	for joint_name in home.keys():
		if String(joint_name) == String(arm.gripper_joint_name):
			continue
		worst = maxf(worst, absf(
			float(home[joint_name]) - robot.get_joint_position(String(joint_name))))
	return worst


## Costo per passo in [-1, 0] per arrivare a casa senza fermarsi.
##
## Attivo solo dopo la consegna e solo VICINO a casa: durante il viaggio la velocita' e' quello che
## serve, ed e' fermandosi sul posto che il compito si chiude.
## PREMIO, non costo. La versione a costo ha rotto v55: dopo la consegna il braccio pagava a ogni
## passo finche' non si fermava esattamente a casa, e su ~150 passi di coda il conto superava i +30
## della consegna stessa. Una policy che ancora non sa fermarsi ha quindi smesso di consegnare --
## misurato, consegne da 17 a 0 mentre le prese salivano a 128 su 150. Rendere negativa una fase
## intera e' un invito a non entrarci.
##
## In premio non puo' succedere: la coda vale al massimo zero, quindi consegnare resta conveniente e
## fermarsi a casa e' un extra.
func get_home_still_reward() -> float:
	if not _delivered or _home_pose == null or arm.tool_pose == null:
		return 0.0
	var away := (arm.tool_pose as Node3D).global_position.distance_to(_home_pose.global_position)
	# STRETTO, non sulla banda intera. Con la vicinanza calcolata sui 300 mm del viaggio, stare fermo
	# a 70 mm da casa valeva gia' 0,145 per passo -- e misurato, la policy si fermava esattamente li',
	# a 64-70 mm, a velocita' 0,20. Il premio pagava il sostare vicino invece dell'arrivare. Ora la
	# finestra e' il doppio del gate: fuori di li' non c'e' niente da incassare, e l'unico modo di
	# guadagnare e' entrare davvero nel gate.
	var window := maxf(success_distance * 2.0, 0.001)
	var nearness := clampf(1.0 - away / window, 0.0, 1.0)
	if nearness <= 0.0:
		return 0.0
	var speed := float(arm.call("_max_joint_speed"))
	var stillness := 1.0 - clampf(speed / maxf(home_still_scale, 0.0001), 0.0, 1.0)
	return nearness * stillness


func get_claw_floor_penalty() -> float:
	if _finger_tips.size() < 2:
		return 0.0
	var lowest := minf(_tip_world(0).y, _tip_world(1).y)
	var depth := (_surface_top_y - maxf(claw_floor_free_margin, 0.0)) - lowest
	if depth <= 0.0:
		return 0.0
	return -clampf(depth / maxf(claw_floor_penalty_scale, 0.0001), 0.0, 1.0)


## Per-step cost in [-1, 0] for approaching with the claws already shut.
##
## The OpenArm charges this against the measured pad gap; here there is no usable pad geometry, so
## it is charged against the commanded closure instead. Same purpose: arriving clenched means the
## object cannot enter the hand, and without a cost that state is free -- on the OpenArm the policy
## found it and spent whole episodes shut, which made the grasp impossible from that step onwards.
## Zero once holding, and zero once the tool is at the object, where closing is the right move.
##
## Faded rather than switched. As a hard step at the capture radius it charged full price for closing
## anywhere outside it: measured over 3300 episodes the policy settled about 6 cm out, which is inside
## the penalty zone, so every attempt to close cost up to -30 across an episode -- the same size as
## the goal reward it was trying to earn. The grasp branch was never sampled once, so the critic never
## saw what a grasp is worth. Fading leaves a band where trying is cheap.
func get_premature_closure_penalty() -> float:
	if grasp_target == null or arm.is_grasp_attached():
		return 0.0
	var tool: Node3D = arm.tool_pose
	if tool == null:
		return 0.0
	# Free only once the cube is within the claws' own closing travel of the grasp point. Asked for
	# directly: closing should not be encouraged until the object is at least as close as the distance
	# the claws themselves sweep. That travel is measured, not assumed -- the tips advance about 23 mm
	# between open and shut -- so the free radius is the larger of the capture distance and the sweep,
	# and it means the same thing if either is retuned.
	var capture := maxf(maxf(grasp_capture_distance, 0.0), _closing_sweep())
	var distance := tool.global_position.distance_to(grasp_target.global_position)
	if distance <= capture:
		return 0.0
	var full := maxf(grasp_premature_closure_full_distance, capture + 0.0001)
	var fade := clampf((distance - capture) / (full - capture), 0.0, 1.0)
	return -_gripper_closure() * fade


## Tilt of the object away from upright, in degrees.
func grasp_object_tilt_degrees() -> float:
	if grasp_target == null:
		return 0.0
	var up := grasp_target.global_basis.orthonormalized() * Vector3.UP
	return rad_to_deg(up.angle_to(Vector3.UP))


## Measures the object and the surface it rests on, instead of trusting hand-written metres.
## Rimette i due array che il salvataggio dall'editor svuota.
##
## `gripper_finger_link_names` e `environment_collision_exempt_links` stanno sul nodo RobotArm, che e'
## un'istanza di `x_arm_grasp_agent.tscn`. Ogni volta che la scena dello scenario viene salvata
## dall'editor, Godot ci scrive sopra un override vuoto -- cinque volte finora -- e non e' un errore
## che si veda: senza le punte calibrate il punto utensile ripiega 50 mm piu' indietro, ogni piolo del
## curriculum viene scartato e le chele diventano corpi come gli altri, quindi la presa e' impossibile.
##
## Invece di ricordarsi di rimetterli a mano, la scena li rimette da sola all'avvio. Lo dice, pero':
## un ripristino silenzioso nasconderebbe che il file su disco e' ancora sbagliato, e il prossimo che
## apre la scena nell'editor la salva di nuovo cosi'.
func _restore_claw_links() -> void:
	if arm == null:
		return
	if arm.gripper_finger_link_names.is_empty():
		arm.gripper_finger_link_names = claw_link_names.duplicate()
		push_warning("[contratto] gripper_finger_link_names era vuoto: rimesso dalla scena (%s). "
			% [str(claw_link_names)] + "Il file .tscn e' ancora da sistemare.")
		print("[contratto] gripper_finger_link_names era vuoto: rimesso dalla scena")
	if arm.environment_collision_exempt_links.is_empty():
		arm.environment_collision_exempt_links = claw_link_names.duplicate()
		push_warning("[contratto] environment_collision_exempt_links era vuoto: rimesso dalla scena")
		print("[contratto] environment_collision_exempt_links era vuoto: rimesso dalla scena")


func _calibrate() -> void:
	_object_half_extents = _shape_half_extents(grasp_target)
	var surface: Node3D = spawn_surface
	if surface == null:
		surface = get_node_or_null("Environment/Floor") as Node3D
	if surface != null:
		_surface_top_y = _shape_top_y(surface)
	_calibrated = _object_half_extents != Vector3.ZERO
	print("[xarm_grasp_zone] object half extents %s | surface top y=%.4f | resting y=%.4f" % [
		_object_half_extents, _surface_top_y, _surface_top_y + _object_half_extents.y])
	if spawn_area != null:
		var bounds := _spawn_bounds()
		print("[xarm_grasp_zone] spawn x=[%.4f, %.4f] z=[%.4f, %.4f]" % [
			bounds[0].x, bounds[1].x, bounds[0].z, bounds[1].z])


func _shape_half_extents(root: Node3D) -> Vector3:
	if root == null:
		return Vector3.ZERO
	for child in root.get_children():
		var collider := child as CollisionShape3D
		if collider == null or collider.shape == null:
			continue
		var scale := collider.global_basis.get_scale().abs()
		if collider.shape is BoxShape3D:
			return (collider.shape as BoxShape3D).size * 0.5 * scale
		if collider.shape is SphereShape3D:
			var r := (collider.shape as SphereShape3D).radius
			return Vector3(r, r, r) * scale
		if collider.shape is CylinderShape3D:
			var cylinder := collider.shape as CylinderShape3D
			return Vector3(
				cylinder.radius, cylinder.height * 0.5, cylinder.radius) * scale
	return Vector3.ZERO


## Highest world point of the surface's collision shape.
##
## Not `body.global_position.y + half_extent`: that assumes the shape is centred on its body, and it
## is not. The floor slab is 200 mm thick with its shape pushed 100 mm down so its FACE stays at
## y=0 -- the thickness is there because a 1 mm plate let the cube be shoved straight through it --
## and the centred assumption read the surface 100 mm too high, floating the cube in mid-air.
func _shape_top_y(root: Node3D) -> float:
	if root == null:
		return 0.0
	var top := -INF
	for child in root.get_children():
		var collider := child as CollisionShape3D
		if collider == null or collider.shape == null:
			continue
		var box := collider.shape.get_debug_mesh().get_aabb()
		for corner in range(8):
			var local := box.position + box.size * Vector3(
				float(corner & 1), float((corner >> 1) & 1), float((corner >> 2) & 1))
			top = maxf(top, (collider.global_transform * local).y)
	return top if top > -INF else 0.0


## World-space xz bounds of the spawn area, already pulled in by the object's half width so the
## object lands fully inside the plate rather than overhanging it.
func _spawn_bounds() -> Array:
	var centre := spawn_area.global_position
	var half := _shape_half_extents(spawn_area)
	var inset := Vector3(
		maxf(_object_half_extents.x, 0.0) + maxf(spawn_edge_margin, 0.0),
		0.0,
		maxf(_object_half_extents.z, 0.0) + maxf(spawn_edge_margin, 0.0))
	var lower := centre - half + inset
	var upper := centre + half - inset
	# A margin wider than the area itself would invert the range; collapse to the centre instead.
	if lower.x > upper.x:
		lower.x = centre.x
		upper.x = centre.x
	if lower.z > upper.z:
		lower.z = centre.z
		upper.z = centre.z
	return [lower, upper]


func _on_scenario_configured(config: Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))
	_training_mode = bool(config.get("training_mode", controller.training_mode))
	# `curriculum_level` was being IGNORED, silently. run.py sends it whenever --curriculum-level is
	# given, and every evaluation that thought it was pinning a rung was in fact running whatever
	# level that process had accumulated from its own captures -- so an evaluation asking for the
	# hardest rung could be measuring the easiest one. The level is the reverse curriculum's own:
	# 0 seeds the hand ON the cube, 1 seeds it `reverse_standoff_far` above it.
	if config.has("curriculum_level"):
		_reverse_level = clampf(float(config["curriculum_level"]), 0.0, 1.0)
	# Neither rung is the home start: at full retreat the hand still begins 135 mm above the cube.
	# Turning the curriculum off is what starts an episode from home, and an evaluation has to be
	# able to ask for that -- otherwise "solved" means solved from above the cube.
	if config.has("reverse_curriculum"):
		reverse_curriculum_enabled = bool(config["reverse_curriculum"])


func _on_episode_reset_started(seed_value: int) -> void:
	if grasp_target == null:
		return
	# The episode that just ENDED, one JSON line, before anything is cleared. Written only when
	# XARM_EPISODE_LOG names a file, so it costs nothing in training.
	#
	# It exists because the same question has been answered wrongly twice from aggregate numbers: a
	# 95% success rate that was measuring arrival instead of delivery, and a 100% that was measuring
	# the curriculum's easiest rung instead of the task. A per-episode record of WHERE the cube was
	# and WHICH link of the chain broke cannot be read that way.
	_log_finished_episode()
	_episode_rng.seed = seed_value
	arm.success_distance = success_distance
	arm.success_hold_physics_frames = success_hold_physics_frames
	arm.success_max_joint_speed = success_max_joint_speed
	# With orientation out of the task, the angle gate is opened all the way and the shaped
	# orientation term is zeroed, so nothing scores the wrist's roll. Applied here at every reset
	# rather than in the agent scene, so the two always agree and the flag is the single switch.
	arm.success_angle_degrees = success_angle_degrees if orientation_matters else 180.0
	arm.orientation_progress_weight = arm.orientation_progress_weight if orientation_matters else 0.0
	target_yaw_follows_azimuth = target_yaw_follows_azimuth and orientation_matters
	if log_capture_rejections and _capture_attempts > 0:
		print(("[cattura] tentativi %d | rifiuti: fra-le-punte %d, lato %d, profondita %d, laterale %d, varco %d, moto %d"
			+ " | migliori: lato %+.4f profondita %.4f laterale %.4f varco %+.4f moto %.3f") % [
			_capture_attempts, _reject_between, _reject_across, _reject_depth, _reject_side,
			_reject_gap, _reject_speed,
			_best_across, _best_depth, _best_side, _best_gap, _best_speed])
		# The rung the hand is being seeded on. Zero means the seed puts it ON the cube, which is
		# where it stays until a capture pays for a retreat -- silent otherwise, and easy to mistake
		# for a scene that simply never starts from home.
		print("[cattura] curriculum inverso: livello %.3f, distacco %.1f mm%s" % [
			_reverse_level, reverse_standoff_for_episode(_training_episode) * 1000.0,
			("  <-- partito DA CASA" if _home_start_episode else "")])
		if _grip_command_count > 0:
			print("[pinza] comando: medio %+.3f, piu' chiuso %+.3f | chiusura raggiunta %.3f" % [
				_grip_command_sum / float(_grip_command_count), _grip_command_min,
				_grip_closure_max])
		if _carry_highest > -INF:
			print("[trasporto] cubo alzato al massimo a %.1f mm (a riposo %.1f)" % [
				_carry_highest * 1000.0, (_surface_top_y + _object_half_extents.y) * 1000.0])
		if _tip_lowest < INF:
			print("[chele] punta piu' bassa %.1f mm (pavimento a %.1f, cubo a riposo %.1f)" % [
				_tip_lowest * 1000.0, _surface_top_y * 1000.0,
				(_surface_top_y + _object_half_extents.y) * 1000.0])
		if _cube_lowest < INF:
			print("[cubo] piu' in basso %.1f mm (a riposo %.1f) | spostato %.1f mm | girato %.1f gradi" % [
				_cube_lowest * 1000.0, _cube_spawn.y * 1000.0, _cube_max_shift * 1000.0,
				_cube_max_turn])
			if _cube_speed_clamps > 0 or _cube_rescues > 0 or _cube_pin_pushes > 0:
				print("[cubo] frenato %d | rimesso sul piano %d | rimesso sullo spawn %d volte" % [
					_cube_speed_clamps, _cube_rescues, _cube_pin_pushes])
	_capture_attempts = 0
	_cube_lowest = INF
	_tip_lowest = INF
	_carry_highest = -INF
	_floor_breached = false
	_breach_depth_max = 0.0
	_breach_source = ""
	_cube_max_shift = 0.0
	_cube_max_turn = 0.0
	_grip_command_min = INF
	_grip_command_sum = 0.0
	_grip_command_count = 0
	_grip_closure_max = 0.0
	_cube_speed_clamps = 0
	_cube_rescues = 0
	_reject_across = 0
	_reject_between = 0
	_reject_reach = 0
	_reject_depth = 0
	_reject_side = 0
	_reject_gap = 0
	_reject_speed = 0
	_best_across = INF
	_best_between = INF
	_best_reach = INF
	_best_depth = INF
	_best_side = INF
	_best_gap = INF
	_best_speed = INF
	_carrying = false
	_delivered = false
	_zone_crossed = false
	_crossing_offset = Vector2.ZERO
	_crossing_reward_pending = 0.0
	_credited_capture = false
	_home_closest = INF
	_home_pose_best = INF
	_gate_success_distance = -1.0
	_gate_success_target = ""
	_place_target(_sample_spawn_position(_episode_rng))
	# Read back where the cube actually landed, not where it was asked to go: the sampled point
	# carries no height, so measuring drift against it would report the cube's own resting height as
	# 15 mm of drift in every episode.
	_cube_spawn = grasp_target.global_position
	_cube_spawn_basis = grasp_target.global_basis
	_cube_last_seen = _cube_spawn
	_final_approach_last = INF
	# Locks back on for the next approach: they are an approach-time simplification, not a property of
	# the object.
	if cube_free_after_release:
		grasp_target.axis_lock_angular_x = true
		grasp_target.axis_lock_angular_y = true
		grasp_target.axis_lock_angular_z = true
	_cube_pin_left = cube_pin_frames
	_goal_paid = false
	_lifted = false
	_cube_pin_pushes = 0
	if drop_enabled:
		_retarget_to_cube()

	# Rung and jitter go in TOGETHER. They share one channel -- set_reset_joint_offsets replaces
	# whatever was there -- so calling the seeding first and the jitter second silently threw the
	# rung away, and the probe measured the hand still at home, 443 mm from every standoff.
	# A consegna gia' fatta il curriculum inverso non c'entra: il compito comincia sopra la zona.
	var offsets := _seed_delivered_state() if start_after_delivery else _seed_reverse_curriculum()
	var jitter := _sample_joint_offsets(_episode_rng, joint_jitter_degrees)
	if offsets.is_empty():
		offsets = jitter
	else:
		for index in range(mini(offsets.size(), jitter.size())):
			offsets[index] += jitter[index]
	arm.set_reset_joint_offsets(offsets)


func _sample_spawn_position(rng: RandomNumberGenerator) -> Vector3:
	if spawn_area == null or not _calibrated:
		return grasp_target.global_position
	# An explicit spot, when something outside wants to choose it. Used by the demonstration recorder
	# to walk a grid over the spawn area instead of hoping random seeds cover it: a demonstration set
	# is only as good as the spread of positions in it.
	if _spawn_override_active:
		return Vector3(
			_spawn_override.x, _surface_top_y + _object_half_extents.y, _spawn_override.z)
	var bounds := _spawn_bounds()
	var lower: Vector3 = bounds[0]
	var upper: Vector3 = bounds[1]
	return Vector3(
		rng.randf_range(lower.x, upper.x),
		_surface_top_y + _object_half_extents.y,
		rng.randf_range(lower.z, upper.z))


## Forces the next spawns to a chosen spot, or hands sampling back to the episode RNG.
func set_spawn_override(where: Variant) -> void:
	if where == null:
		_spawn_override_active = false
		return
	_spawn_override = where
	_spawn_override_active = true


## The corners of the spawn area, for anything that wants to cover it deliberately.
func spawn_bounds() -> Array:
	return _spawn_bounds()


## Teleports the body and stands it up square, then fixes the orientation the arm is asked for.
func _place_target(position: Vector3) -> void:
	# A new episode cannot start still holding the previous one's object.
	if arm.has_method("is_grasp_attached") and arm.is_grasp_attached():
		_release_object()
	_demanded_basis = _demanded_basis_for(position)
	_demanded_basis_valid = true
	_spawn_position = position

	var resting := Transform3D(_demanded_basis, position)
	var rid := grasp_target.get_rid()
	# Through the physics server, so the reset observation already sees the new pose.
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_TRANSFORM, resting)
	PhysicsServer3D.body_set_state(
		rid, PhysicsServer3D.BODY_STATE_LINEAR_VELOCITY, Vector3.ZERO)
	PhysicsServer3D.body_set_state(
		rid, PhysicsServer3D.BODY_STATE_ANGULAR_VELOCITY, Vector3.ZERO)
	grasp_target.global_transform = resting
	grasp_target.linear_velocity = Vector3.ZERO
	grasp_target.angular_velocity = Vector3.ZERO
	grasp_target_pose.global_basis = _demanded_basis


func _demanded_basis_for(position: Vector3) -> Basis:
	var basis := Basis.IDENTITY
	if target_yaw_follows_azimuth:
		basis = Basis(Vector3.UP, _target_azimuth_yaw(position)) * basis
	if (
		target_yaw_randomization_degrees > 0.0
		and _training_episode >= target_yaw_randomization_start_episode
	):
		var jitter := deg_to_rad(_episode_rng.randf_range(
			-target_yaw_randomization_degrees, target_yaw_randomization_degrees))
		basis = Basis(Vector3.UP, jitter) * basis
	return basis.orthonormalized()


## Yaw that turns the demanded forward axis towards the object, seen from the arm base.
##
## The z is negated because Basis(UP, t) * RIGHT == (cos t, 0, -sin t): without it the pose turns
## the wrong way and the error grows with azimuth instead of vanishing.
func _target_azimuth_yaw(target_position: Vector3) -> float:
	if arm == null:
		return 0.0
	var offset := target_position - (arm as Node3D).global_position
	if absf(offset.x) < 0.0001 and absf(offset.z) < 0.0001:
		return 0.0
	return atan2(-offset.z, offset.x) * target_azimuth_yaw_gain


func _sample_joint_offsets(rng: RandomNumberGenerator, max_degrees: float) -> Array[float]:
	var offsets: Array[float] = []
	var joint_count: int = arm.get_joint_count()
	offsets.resize(joint_count)
	for index in range(joint_count):
		offsets[index] = deg_to_rad(rng.randf_range(-max_degrees, max_degrees))
	return offsets


## Once per episode, whatever the retargets do.
##
## The one-shot event rewards are re-armed by `rebase_agent_tracking`, and a missed drop calls it: so
## carry, hold at the drop pose (+30), open the claws wide of the footprint, pick it up and collect
## the +30 again paid better than delivering, where `_delivered` latches and nothing more is earned.
func _on_target_reached() -> void:
	if _goal_paid:
		return
	_goal_paid = true
	goal_event.trigger(str(arm.name))


## Deliberately does nothing.
##
## The agent raises this whenever the demanded pose node moves more than a millimetre, and that node
## is parented to the CUBE -- so every nudge of the cube counted as the scenario having moved the
## goalposts. A rebase resets the stall guard, erases the backward-progress penalty for the shove,
## and re-arms every one-shot event reward, which together made shoving the cube the cheapest way to
## survive the 120-step stall window. Measured across v3-v12: the cube was pushed more than 20 mm in
## 18-63% of episodes.
##
## The two moves the scenario really makes -- retargeting to the drop pose on capture, and back to
## the cube on a missed drop -- rebase explicitly where they happen, so nothing is lost here.
func _on_target_pose_relocated() -> void:
	pass


func _on_obstacle_collision() -> void:
	collision_event.trigger(str(arm.name))
