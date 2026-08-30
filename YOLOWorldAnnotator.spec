# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, get_package_paths


project_root = Path(SPECPATH)
source_root = project_root / "src"
ultra_datas, ultra_binaries, ultra_hiddenimports = collect_all("ultralytics")
clip_datas = collect_data_files("clip")

qt_runtime_binaries = []
if sys.platform == "win32":
    pyside_dir = Path(get_package_paths("PySide6")[1])
    shiboken_dir = Path(get_package_paths("shiboken6")[1])
    for candidate in (
        shiboken_dir / "shiboken6.abi3.dll",
        pyside_dir / "concrt140.dll",
        pyside_dir / "msvcp140_codecvt_ids.dll",
    ):
        if candidate.is_file():
            qt_runtime_binaries.append((str(candidate), "PySide6"))

a = Analysis(
    [str(source_root / "yolo_world_annotator" / "__main__.py")],
    pathex=[str(source_root)],
    binaries=[*ultra_binaries, *qt_runtime_binaries],
    datas=[*ultra_datas, *clip_datas],
    hiddenimports=ultra_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "IPython", "jupyter", "notebook"],
    noarchive=False,
    optimize=0,
)

if sys.platform == "win32":
    conflicting_icu = {"icuuc.dll", "icudt73.dll"}
    a.binaries = type(a.binaries)(
        item for item in a.binaries if Path(item[0]).name.lower() not in conflicting_icu
    )

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="YOLOWorldAnnotator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="YOLOWorldAnnotator",
)
