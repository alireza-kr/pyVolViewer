"""
Dash layout pieces. Each module builds one region of the page as plain
``html``/``dcc`` components; :mod:`interactive_glm.components.layout`
assembles them into the full app layout.

Nothing here registers callbacks - see :mod:`interactive_glm.callbacks` for
the interactive behaviour. Keeping layout and callbacks in separate
modules means you can change how something looks without needing to
re-read how it behaves, and vice versa.
"""
