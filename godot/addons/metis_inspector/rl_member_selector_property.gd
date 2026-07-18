@tool
extends EditorProperty

enum SelectorMode {
	METHOD,
	PROPERTY
}

const SUPPORTED_TYPES := [
	TYPE_BOOL,
	TYPE_INT,
	TYPE_FLOAT,
	TYPE_VECTOR2,
	TYPE_VECTOR3,
	TYPE_ARRAY,
	TYPE_PACKED_FLOAT32_ARRAY,
	TYPE_PACKED_FLOAT64_ARRAY,
	TYPE_PACKED_INT32_ARRAY,
	TYPE_PACKED_INT64_ARRAY
]

var _mode: SelectorMode = SelectorMode.METHOD
var _line_edit := LineEdit.new()
var _menu_button := MenuButton.new()
var _updating := false


func _init() -> void:
	var row := HBoxContainer.new()
	row.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	_line_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_line_edit.placeholder_text = "Method name" if _mode == SelectorMode.METHOD else "Property path"
	_line_edit.text_submitted.connect(_on_text_submitted)
	_line_edit.focus_exited.connect(_on_focus_exited)
	row.add_child(_line_edit)

	_menu_button.text = "Select..."
	_menu_button.tooltip_text = "Choose a compatible member from the selected source node"
	_menu_button.get_popup().about_to_popup.connect(_rebuild_menu)
	_menu_button.get_popup().id_pressed.connect(_on_member_selected)
	row.add_child(_menu_button)

	add_child(row)
	add_focusable(_line_edit)


func configure(mode:SelectorMode) -> void:
	_mode = mode
	_line_edit.placeholder_text = "Method name" if _mode == SelectorMode.METHOD else "Property path"


func _update_property() -> void:
	var edited_object := get_edited_object()
	if edited_object == null:
		return
	_updating = true
	_line_edit.text = str(edited_object.get(get_edited_property()))
	_updating = false


func _rebuild_menu() -> void:
	var popup := _menu_button.get_popup()
	popup.clear()
	var edited_object := get_edited_object()
	var target := _resolve_target(edited_object)
	if target == null:
		popup.add_item("No source node available")
		popup.set_item_disabled(0, true)
		return

	var members := _collect_methods(target, edited_object) if _mode == SelectorMode.METHOD else _collect_properties(target)
	if members.is_empty():
		popup.add_item("No compatible members")
		popup.set_item_disabled(0, true)
		return
	for member in members:
		var item_id := popup.item_count
		popup.add_item(str(member), item_id)
		popup.set_item_metadata(item_id, str(member))


func _resolve_target(edited_object:Object) -> Node:
	if edited_object != null and edited_object.has_method("get_editor_source"):
		var target: Variant = edited_object.call("get_editor_source")
		if target is Node:
			return target
	return null


func _collect_methods(target:Node, edited_object:Object) -> Array[String]:
	var result: Array[String] = []
	var script := target.get_script()
	var methods: Array = script.get_script_method_list() if script != null else target.get_method_list()
	var has_bound_argument := not str(edited_object.get("bind_string_arg")).is_empty()
	for method in methods:
		var method_name := str(method.get("name", ""))
		if method_name.is_empty() or method_name.begins_with("_"):
			continue
		var arguments: Array = method.get("args", [])
		var default_arguments: Array = method.get("default_args", [])
		var required_count := maxi(arguments.size() - default_arguments.size(), 0)
		if has_bound_argument:
			if arguments.size() < 1 or required_count > 1:
				continue
		elif required_count > 0:
			continue
		var return_info: Dictionary = method.get("return", {})
		var return_type := int(return_info.get("type", TYPE_NIL))
		if return_type != TYPE_NIL and return_type not in SUPPORTED_TYPES:
			continue
		result.append(method_name)
	result.sort()
	return result


func _collect_properties(target:Node) -> Array[String]:
	var result: Array[String] = []
	for property in target.get_property_list():
		var property_name := str(property.get("name", ""))
		var property_type := int(property.get("type", TYPE_NIL))
		if property_name.is_empty() or property_name.begins_with("_") or "/" in property_name:
			continue
		if property_type not in SUPPORTED_TYPES:
			continue
		result.append(property_name)
	result.sort()
	return result


func _on_member_selected(item_id:int) -> void:
	var popup := _menu_button.get_popup()
	var value := str(popup.get_item_metadata(popup.get_item_index(item_id)))
	_line_edit.text = value
	_commit_value(value)


func _on_text_submitted(value:String) -> void:
	_commit_value(value)


func _on_focus_exited() -> void:
	if not _updating:
		_commit_value(_line_edit.text)


func _commit_value(value:String) -> void:
	if _updating or get_edited_object() == null:
		return
	var stored_value: Variant = NodePath(value) if _mode == SelectorMode.PROPERTY else StringName(value)
	emit_changed(get_edited_property(), stored_value)
