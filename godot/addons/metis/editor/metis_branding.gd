@tool
extends RefCounted
## Shared Metis branding for editor dialogs and the toolbar.
##
## The logo lives at two different paths depending on how Metis is being used, so every call site
## needs the same fallback: the packaged add-on ships `addons/metis/logo.svg` (build_release.py
## copies godot/icon.svg there), while the development project has only `res://icon.svg`. Loading it
## in one place keeps the toolbar and the dialogs from drifting apart.

const PACKAGED_LOGO := "res://addons/metis/logo.svg"
const FALLBACK_LOGO := "res://icon.svg"
const SUBTITLE_NAME := "MetisHeaderSubtitle"


static func logo() -> Texture2D:
	for path in [PACKAGED_LOGO, FALLBACK_LOGO]:
		if ResourceLoader.exists(path):
			return load(path)
	return null


static func logo_rect(size: int) -> TextureRect:
	## A TextureRect that renders the logo at `size` px regardless of the source resolution.
	##
	## EXPAND_IGNORE_SIZE matters: the SVG rasterises to 576px, and without it that becomes the
	## control's minimum size, which is what blew the editor's top bar apart the first time.
	var rect := TextureRect.new()
	rect.texture = logo()
	rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	rect.custom_minimum_size = Vector2(size, size)
	rect.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	rect.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	return rect


static func header(title_text: String, subtitle_text: String = "") -> Control:
	## Logo + title (+ optional subtitle) banner for the top of a dialog, followed by a separator.
	##
	## An AcceptDialog's `title` is the editor window's titlebar and cannot hold a texture, so the
	## brand has to live in the dialog body instead.
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
	# Follow the editor theme instead of hardcoding a size, so the header matches whatever font
	# scale and colour scheme the user runs.
	title.theme_type_variation = "HeaderMedium"
	text.add_child(title)

	if subtitle_text != "":
		var subtitle := Label.new()
		# Named so a caller whose subtitle changes at runtime can find it with
		# find_child(SUBTITLE_NAME, true, false) instead of relying on the child order here.
		subtitle.name = SUBTITLE_NAME
		subtitle.text = subtitle_text
		subtitle.theme_type_variation = "HeaderSmall"
		subtitle.modulate = Color(1.0, 1.0, 1.0, 0.65)
		text.add_child(subtitle)

	row.add_child(text)
	box.add_child(row)
	box.add_child(HSeparator.new())
	return box
