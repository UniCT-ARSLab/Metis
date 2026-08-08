@tool
extends RefCounted
## Tells a Metis scenario scene from any other scene, without instantiating it.
##
## A trainable/runnable scenario is a scene holding a BridgeServer: that node is the TCP endpoint the
## Python side connects to, so a scene without one cannot be driven whatever it is named. Of the 14
## scenes under `scenarios/` in the development project, 9 qualify -- the rest are sub-scenes (a ball,
## a missile, map variants) that nothing should ever be pointed at.

# Referenced by string rather than by the BridgeServer symbol, so a caller still parses if the
# runtime scripts fail to load.
const BRIDGE_CLASS := "BridgeServer"

# Depth cap for the recursion below: a guard against a malformed cyclic instance chain, not a case
# any real project is expected to hit.
const MAX_DEPTH := 8


static func has_bridge_server(path: String, depth: int = 0) -> bool:
	## Whether the scene FILE at `path` contains a BridgeServer.
	##
	## SceneState exposes the saved node list, so this costs a resource load and runs no _ready() --
	## instantiating a scenario in the editor would start its @tool scripts and build its whole tree.
	## The file is also the right thing to inspect rather than the live edited tree: launching starts
	## a separate Godot that loads this path from disk, so unsaved edits are not what will run.
	##
	## Recurses into instanced sub-scenes, because an inheriting scenario holds no BridgeServer of its
	## own: cars_path_aware_scenario.tscn instances cars_scenario.tscn and gets it from there, which a
	## plain text scan of the .tscn misses.
	if depth > MAX_DEPTH or path.is_empty() or not ResourceLoader.exists(path):
		return false
	var packed := load(path) as PackedScene
	if packed == null:
		return false
	var state := packed.get_state()
	for node_index in state.get_node_count():
		var nested := state.get_node_instance(node_index)
		if nested != null and has_bridge_server(nested.resource_path, depth + 1):
			return true
		for property_index in state.get_node_property_count(node_index):
			if state.get_node_property_name(node_index, property_index) != "script":
				continue
			var script := state.get_node_property_value(node_index, property_index) as Script
			if script != null and script.get_global_name() == BRIDGE_CLASS:
				return true
	return false


static func open_scenario_path() -> String:
	## The scene currently open in the editor, if it is a scenario; "" otherwise.
	var root := EditorInterface.get_edited_scene_root()
	var path := "" if root == null else root.scene_file_path
	return path if has_bridge_server(path) else ""
