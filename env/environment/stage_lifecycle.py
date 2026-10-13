"""Replace stages without leaving queued Action Graph work on deleted prims."""


def replace_current_stage(context, app) -> None:
    """Flush Kit stage events before closing and after replacing the stage."""
    app.update()
    context.close_stage()
    app.update()
    context.new_stage()
    app.update()
