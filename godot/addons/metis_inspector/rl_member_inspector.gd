@tool
extends EditorInspectorPlugin

const MemberSelectorProperty = preload("res://addons/metis_inspector/rl_member_selector_property.gd")


func _can_handle(object:Object) -> bool:
	return object is MethodObservationSource or object is PropertyObservationSource


func _parse_property(
	object:Object,
	_type:Variant.Type,
	name:String,
	_hint_type:PropertyHint,
	_hint_string:String,
	_usage_flags:PropertyUsageFlags,
	_wide:bool
) -> bool:
	if object is MethodObservationSource and name == "method_name":
		var method_selector := MemberSelectorProperty.new()
		method_selector.configure(MemberSelectorProperty.SelectorMode.METHOD)
		add_property_editor(name, method_selector)
		return true
	if object is PropertyObservationSource and name == "property_path":
		var property_selector := MemberSelectorProperty.new()
		property_selector.configure(MemberSelectorProperty.SelectorMode.PROPERTY)
		add_property_editor(name, property_selector)
		return true
	return false
