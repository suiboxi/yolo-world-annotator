# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, get_package_paths


project_root = Path(SPECPATH)
ultra_datas, ultra_binaries, ultra_hiddenimports = collect_all("ultralytics")
clip_datas = collect_data_files("clip")
pyside_dir = Path(get_package_paths("PySide6")[1])
shiboken_dir = Path(get_package_paths("shiboken6")[1])

# PySide6 6.11 needs these DLLs on its own Windows DLL search path.
qt_runtime_binaries = [
    (str(shiboken_dir / "shiboken6.abi3.dll"), "PySide6"),
    (str(pyside_dir / "concrt140.dll"), "PySide6"),
    (str(pyside_dir / "msvcp140_codecvt_ids.dll"), "PySide6"),
]

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
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

# Conda's ICU 73 exports a different ABI from the Windows ICU expected by
# PySide6 6.11. If bundled, QtCore fails with WinError 127. On this Windows 11
# target Qt must use the compatible system ICU instead.
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
