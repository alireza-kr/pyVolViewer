"""
Pure functions that build Plotly figures from already-processed arrays.

Like :mod:`interactive_glm.processing`, these functions don't know about
Dash, callbacks, or where their inputs came from - they take arrays/values
in and return a :class:`plotly.graph_objects.Figure` out, which keeps them
easy to test and easy to reuse (e.g. to export a static figure outside the
app).
"""
