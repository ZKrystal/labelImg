# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['labelImg.py'],
    pathex=['E:\labelimg\labelImg'],              # 项目根，保证 libs 能被找到
    binaries=[],
    datas=[
        ('E:\labelimg\labelImg\libs', 'libs'),                 # 本地包
    ],
    hiddenimports=['libs.resources'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[                            # << 大幅瘦身
        'torch', 'torchvision', 'tensorflow', 'tensorboard', 'keras',
        'jupyter', 'IPython', 'matplotlib', 'seaborn', 'pandas', 'scipy',
        'caffe2', 'pytorch', 'pytest', 'tkinter',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---------- 上面你的代码保持不变 ----------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # ← 关键：不把二进制塞进 exe
    name='labelImg',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

# ↓↓↓ 新增：真正负责“分离”的段落
coll = COLLECT(
    exe,                        # 启动器 exe
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    name='labelImg',            # 输出目录名
)
