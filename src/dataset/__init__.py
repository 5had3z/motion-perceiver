from pathlib import Path

from nvidia.dali import plugin_manager

for plugin_path in Path(__file__).parent.glob("**/*.so"):
    plugin_manager.load_library(str(plugin_path))
    print(f"DALI: loaded plugin - {plugin_path.name}")
