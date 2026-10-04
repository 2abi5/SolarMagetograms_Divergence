# Optional: `source env.sh` before running anything. Keeps library caches inside the
# project, activates ./.venv if it exists, and moves to the project root.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export XDG_CONFIG_HOME="$PROJECT_ROOT/.config"
mkdir -p "$XDG_CACHE_HOME/astropy" "$XDG_CONFIG_HOME/astropy" "$XDG_CONFIG_HOME/sunpy" "$XDG_CACHE_HOME/matplotlib"
export SUNPY_CONFIGDIR="$XDG_CONFIG_HOME/sunpy"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
[ -f "$PROJECT_ROOT/.venv/bin/activate" ] && source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"
