# ── Streamlit Cloud entry point ────────────────────────────
# This file exists so Streamlit Cloud can find the app at
# the repo root. It simply re-exports the real dashboard.
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
_dashboard = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py")
exec(compile(open(_dashboard, encoding="utf-8").read(), _dashboard, "exec"),
     {**globals(), "__file__": _dashboard})
