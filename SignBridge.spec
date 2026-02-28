import os
import mediapipe
from PyInstaller.utils.hooks import collect_all

mediapipe_path = os.path.dirname(mediapipe.__file__)

# Collect all resources for mediapipe
mp_datas, mp_binaries, mp_hiddenimports = collect_all('mediapipe')

datas = [
    ('isl_alphabet_model.pkl', '.'),
    ('dynamic_sign_model.onnx', '.'),
    ('templates', 'templates'),
    ('static', 'static'),
    (mediapipe_path, 'mediapipe') # Force include whole package data
] + mp_datas
binaries = [] + mp_binaries
hiddenimports = [
    'flask_socketio', 'socketio', 'engineio', 'engineio.async_drivers.threading',
    'simple_websocket', 'threading', 'sklearn', 'sklearn.ensemble', 'sklearn.tree',
    'sklearn.utils', 'mediapipe.solutions.hands', 'mediapipe.solutions.drawing_utils',
    'mediapipe.python.solutions.hands', 'mediapipe.python.solutions.drawing_utils'
] + mp_hiddenimports

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SignBridge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
