@tool
extends RefCounted
## Identifies Metis scenarios without instantiating their scene trees.

# String lookup works before the runtime class cache is ready.
const BRIDGE_CLASS := "BridgeServer"

# Protect against malformed cyclic scene dependencies.
const MAX_DEPTH := 8


static func has_bridge_server(path: String, depth: int = 0) -> bool:
	## Checks the saved scene and its instances for a BridgeServer node.
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
