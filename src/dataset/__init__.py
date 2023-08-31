from pathlib import Path
import logging

from nvidia.dali import plugin_manager

for plugin_path in Path(__file__).parent.glob("**/*.so"):
    plugin_manager.load_library(str(plugin_path))
    logging.debug("DALI: loaded plugin - %s", plugin_path.name)
