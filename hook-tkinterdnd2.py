"""Include tkinterdnd2's bundled native TkDnD files in PyInstaller builds."""

from PyInstaller.utils.hooks import collect_data_files


datas = collect_data_files("tkinterdnd2")
