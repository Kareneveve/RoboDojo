"""Restore MiVerse language variants from the evaluated scene layout."""


class LayoutInstructionMixin:
    """Use the saved instruction verbatim for each evaluation environment."""

    refresh_instruction_on_reset = True

    def gen_instruction(self, env_idx):
        layout = self.scene_manager.config.get(f"env{env_idx}")
        instruction = layout.get("instruction") if layout is not None else None
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"Scene layout for env {env_idx} must contain a nonempty "
                "'instruction' string for this language-generalization task. "
                "Regenerate the layout with the language task before evaluation."
            )
        return [instruction]
