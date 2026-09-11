@tool
extends RefCounted
## Shared Metis branding for editor controls.

const PACKAGED_LOGO := "res://addons/metis/logo.svg"
const FALLBACK_LOGO := "res://icon.svg"
const SUBTITLE_NAME := "MetisHeaderSubtitle"


static func logo() -> Texture2D:
	for path in [PACKAGED_LOGO, FALLBACK_LOGO]:
		if ResourceLoader.exists(path):
			return load(path)
	return null


static func logo_rect(size: int) -> TextureRect:
	## Creates a logo control with a fixed display size.
	var rect := TextureRect.new()
	rect.texture = logo()
	rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	rect.custom_minimum_size = Vector2(size, size)
	rect.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	rect.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	return rect


static func header(title_text: String, subtitle_text: String = "") -> Control:
	## Builds the branded header used by Metis dialogs.
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 6)

	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 10)
	row.add_child(logo_rect(32))

	var text := VBoxContainer.new()
	text.add_theme_constant_override("separation", 0)
	text.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	text.size_flags_vertical = Control.SIZE_SHRINK_CENTER

	var title := Label.new()
	title.text = title_text
	title.theme_type_variation = "HeaderMedium"
	text.add_child(title)

	if subtitle_text != "":
		var subtitle := Label.new()
		subtitle.name = SUBTITLE_NAME
		subtitle.text = subtitle_text
		subtitle.theme_type_variation = "HeaderSmall"
		subtitle.modulate = Color(1.0, 1.0, 1.0, 0.65)
		text.add_child(subtitle)

	row.add_child(text)
	box.add_child(row)
	box.add_child(HSeparator.new())
	return box
