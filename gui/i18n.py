"""Minimal two-language string table (English default, Chinese)."""
from __future__ import annotations

_LANG = "en"

S = {
    # window / top bar
    "app_title": ("Video Scanner v{v}", "视频扫描 v{v}"),
    "open_video": ("Open video", "打开视频"),
    "save_dir": ("Save dir", "保存目录"),
    "lang_btn": ("中文", "EN"),
    "theme": ("Theme", "主题"),
    "theme_auto": ("Auto", "跟随系统"),
    "theme_light": ("Light", "浅色"),
    "theme_dark": ("Dark", "深色"),
    "no_video": ("No video loaded — open one or drop a file here", "未加载视频 —— 点击打开或把文件拖进来"),
    "video_info": ("{w}×{h} · {fps:.2f} fps · {n} frames · {codec}", "{w}×{h} · {fps:.2f} fps · {n} 帧 · {codec}"),
    # toolbar
    "tb_pan": ("Pan", "拖动视角"),
    "tb_zoom_in": ("Zoom in", "放大"),
    "tb_zoom_out": ("Zoom out", "缩小"),
    "tb_reset": ("Reset view", "默认视角"),
    "tb_position": ("Scanning position", "设置扫描位置"),
    "tb_mask": ("Mask", "设置 mask"),
    "tb_width": ("Width", "设置宽度"),
    "tb_delete": ("Delete", "删除"),
    "tb_info": ("Info", "说明"),
    "del_position": ("Delete scanning position", "删除扫描位置"),
    "del_mask": ("Delete mask", "删除 mask"),
    "del_width": ("Delete width", "删除宽度"),
    "tip_pan": ("Pan mode: drag to move the view. The mouse wheel zooms in every mode.",
                "拖动视角：左键拖拽平移画面。任何模式下滚轮都可以缩放。"),
    "tip_zoom_in": ("Zoom in around the view centre (×1.25)", "以画面中心放大（×1.25）"),
    "tip_zoom_out": ("Zoom out around the view centre (×1.25)", "以画面中心缩小（×1.25）"),
    "tip_reset": ("Fit the frame to the window", "适应窗口"),
    "tip_position": ("Click on the image to place the scanning position line, drag the line to adjust",
                     "在图上单击放置扫描位置竖线，拖动竖线调整位置"),
    "tip_mask": ("Adjust the mask rectangle by dragging its edges / corners / body",
                 "拖动 mask 矩形的边、角或内部进行调整"),
    "tip_width": ("Manual mode only: drag the right edge of the strip band to set the width (fractional allowed)",
                  "仅手动模式：拖动宽度带的右边缘设置宽度（支持小数）"),
    "tip_delete": ("Delete the element of the current mode", "删除当前模式对应的元素"),
    "tip_info": ("How to use the toolbar", "工具栏使用说明"),
    "info_title": ("Toolbar guide", "工具栏说明"),
    "info_html": (
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<tr><th>Button</th><th>Type</th><th>Behaviour</th></tr>"
        "<tr><td>✋ Pan</td><td>mode (default)</td><td>Drag with the left button to move the view</td></tr>"
        "<tr><td>➕ / ➖ Zoom</td><td>action</td><td>Zoom ×1.25 around the view centre (the mouse wheel zooms at the cursor in every mode)</td></tr>"
        "<tr><td>🔄 Reset view</td><td>action</td><td>Fit the frame to the window</td></tr>"
        "<tr><td>📍 Scanning position</td><td>mode</td><td>Click on the image to place the line; drag the line to adjust</td></tr>"
        "<tr><td>🔲 Mask</td><td>mode</td><td>A default rectangle is placed if none exists; drag edges / corners to resize, drag inside to move</td></tr>"
        "<tr><td>📏 Width</td><td>mode (manual only)</td><td>A 5 px band is placed at the position if none exists; drag its right edge to change the width (fractional, 2 decimals). The view zooms to the position automatically and is restored on exit</td></tr>"
        "<tr><td>🗑 Delete</td><td>action</td><td>Delete the element of the current mode; in pan mode a menu lets you choose</td></tr>"
        "<tr><td>ℹ Info</td><td>action</td><td>Show this toolbar operation guide</td></tr>"
        "</table><br>"
        "<ul>"
        "<li>Outside their edit mode the position line, mask and width band stay visible but cannot be dragged.</li>"
        "<li>The left edge of the width band is the position (the strip is <code>frame[:, pos:pos+w]</code>).</li>"
        "<li>In adaptive mode the band shows (read-only) the width that will be used for the current frame.</li>"
        "<li>The mask marks the moving object: motion detection looks for features <b>inside</b> it, the stabilizer <b>outside</b> it.</li>"
        "<li>Live panorama strip (bottom): drag horizontally to inspect earlier parts, double-click to resume following the newest end.</li>"
        "<li>Shortcuts: ←/→ one frame, Shift+←/→ ten frames, Ctrl+O open a video.</li>"
        "</ul>",
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<tr><th>按钮</th><th>类型</th><th>行为</th></tr>"
        "<tr><td>✋ 拖动视角</td><td>模式（默认）</td><td>左键拖拽平移画面</td></tr>"
        "<tr><td>➕ / ➖ 缩放</td><td>单击动作</td><td>以画面中心为基准，每次缩放 ×1.25（任何模式下滚轮都可在光标处缩放）</td></tr>"
        "<tr><td>🔄 默认视角</td><td>单击动作</td><td>适应窗口</td></tr>"
        "<tr><td>📍 设置扫描位置</td><td>模式</td><td>在图上单击放置竖线；之后拖动竖线调整位置</td></tr>"
        "<tr><td>🔲 mask</td><td>模式</td><td>进入时若还没有 mask，就在画面中央放一个默认矩形；拖动边、角缩放，拖动内部平移</td></tr>"
        "<tr><td>📏 宽度</td><td>模式（仅 manual）</td><td>若还没有宽度带，则在 position 处放一个 5 px 的带；拖动右边缘调宽度（支持 2 位小数）。进入时自动放大到 position 附近，退出时恢复原视角</td></tr>"
        "<tr><td>🗑 删除</td><td>单击动作</td><td>删除当前模式对应的元素；在拖动视角模式下点击会弹出菜单让你选择</td></tr>"
        "<tr><td>ℹ 说明</td><td>单击动作</td><td>显示本工具栏操作说明</td></tr>"
        "</table><br>"
        "<ul>"
        "<li>退出编辑模式后，position、mask、宽度带仍然显示，但不能拖动。</li>"
        "<li>宽度带的左边缘就是 position（条带取 <code>frame[:, pos:pos+w]</code>）。</li>"
        "<li>adaptive 模式下，宽度带只读显示当前帧将使用的宽度。</li>"
        "<li>mask 圈出运动物体：运动检测在框<b>内</b>找特征，稳定器在框<b>外</b>找特征。</li>"
        "<li>底部实时长图滚动带：左右拖拽可回看已拼接部分，双击恢复自动跟随最新端。</li>"
        "<li>快捷键：←/→ 逐帧，Shift+←/→ 跳 10 帧，Ctrl+O 打开视频。</li>"
        "</ul>"),
    # panels
    "grp_decoder": ("Decoder", "解码"),
    "probing": ("probing…", "检测中…"),
    "probe_hw_ok": ("✔ hardware decoding · {fps:.0f} fps", "✔ 硬解已启用 · 实测 {fps:.0f} fps"),
    "probe_sw": ("✔ software decoding · {fps:.0f} fps", "✔ 软解 · 实测 {fps:.0f} fps"),
    "probe_fallback": ("✘ not supported for this video, falls back to software · {fps:.0f} fps",
                       "✘ 该设备不支持此视频，已降级为软解 · 实测 {fps:.0f} fps"),
    "probe_fail": ("✘ cannot decode: {msg}", "✘ 无法解码：{msg}"),
    "probe_novideo": ("open a video to test the decoder", "打开视频后检测解码器"),
    "grp_scan": ("Scan parameters", "扫描参数"),
    "position": ("Scanning position", "扫描位置"),
    "manual": ("Manual", "手动"),
    "adaptive": ("Adaptive", "自适应"),
    "width": ("Strip width", "条带宽度"),
    "dir_l2r": ("Object moves left → right", "物体从左向右移动"),
    "dir_r2l": ("Object moves right → left", "物体从右向左移动"),
    "start_frame": ("Detection start frame", "检测起始帧"),
    "use_cur_frame": ("Use current", "设为当前帧"),
    "use_cur_frame_tip": ("Set detection start frame to the current preview frame",
                          "将预览窗口当前的帧设为检测起始帧"),
    "sensitivity": ("Samples", "采样数"),
    "calc_move": ("Calculate move", "计算运动"),
    "mask_lbl": ("Mask (moving object area)", "mask（运动物体区域）"),
    "mask_tip": ("Motion detection searches features INSIDE the mask; the stabilizer searches OUTSIDE it.",
                 "运动检测在 mask 内找特征；稳定器在 mask 外找特征。"),
    "mask_none": ("no mask (whole frame)", "无 mask（整帧）"),
    "grp_stab": ("Stabilizer", "稳定器"),
    "stab_enable": ("Enable stabilizer", "启用稳定器"),
    "stab_mask_hint": (
        "Note: It is recommended to draw a mask around moving objects so camera shake is estimated only from the background outside the mask, avoiding motion errors.",
        "提示：启用稳定器时，建议添加 mask 框住运动物体；程序将仅用 mask 以外的背景区域计算相机视角抖动，避免被运动物体干扰导致运动计算出错。",
    ),
    "detector": ("Detector", "检测器"),
    "model": ("Model", "变换模型"),
    "model_translation": ("Translation (2 DOF)", "平移 (2DOF)"),
    "model_similarity": ("Similarity (4 DOF)", "相似 (4DOF)"),
    "model_homography": ("Homography (8 DOF)", "单应 (8DOF)"),
    "workers": ("Threads (0 = auto)", "线程数（0 = 自动）"),
    "grp_segments": ("Segments (adaptive)", "分段（自适应）"),
    "seg_start": ("Start frame", "起始帧"),
    "seg_width": ("Width", "宽度"),
    "seg_add": ("Add row", "添加行"),
    "seg_del": ("Remove row", "删除行"),
    "grp_output": ("Output", "输出"),
    "split_width": ("Tile width (px)", "图块宽度"),
    "interp": ("Interpolation", "插值方式"),
    "feather": ("Seam feather (px, 0 = off)", "接缝羽化 (px，0 = 关闭)"),
    "start": ("Start", "开始处理"),
    "stop": ("Stop", "停止"),
    # status
    "st_ready": ("Ready", "就绪"),
    "st_preparing": ("Preparing…", "准备中…"),
    "st_processing": ("{done}/{total} frames · {fps:.1f} fps · ETA {eta} · tiles saved {tiles}",
                      "{done}/{total} 帧 · {fps:.1f} fps · 剩余 {eta} · 已保存 tile {tiles}"),
    "st_finalizing": ("Saving the last tile…", "正在保存最后一个 tile…"),
    "st_done": ("Done: {tiles} tiles, {cols} px wide, {sec:.1f} s ({fps:.1f} fps)",
                "完成：{tiles} 张 tile，共 {cols} px，耗时 {sec:.1f} 秒（{fps:.1f} fps）"),
    "st_stopped": ("Stopped: saved {tiles} tiles ({cols} px)", "已停止：已保存 {tiles} 张 tile（共 {cols} px）"),
    "st_error": ("Error — see the message", "出错 —— 请查看提示"),
    "st_motion": ("Calculating movement… {i}/{n}", "计算运动中… {i}/{n}"),
    "st_motion_done": ("↑ Segments calculated (editable){warn}", "↑ 分段已计算（可编辑）{warn}"),
    "st_motion_warn": (" · {n} samples failed and reuse the previous width", " · {n} 个采样失败，沿用前一个宽度"),
    "hover": ("x={x}  y={y}  RGB({r},{g},{b})", "x={x}  y={y}  RGB({r},{g},{b})"),
    "live": ("LIVE · frame {f}", "实时 · 第 {f} 帧"),
    # dialogs
    "dlg_open": ("Choose a video", "选择视频"),
    "dlg_save": ("Choose the save directory", "选择保存目录"),
    "dlg_conflict_title": ("Existing output", "已有输出"),
    "dlg_conflict": ("The output folder already contains {n} file(s) from a previous run:\n{dir}\n\nWhat do you want to do?",
                     "输出目录中已有上次运行留下的 {n} 个文件：\n{dir}\n\n要怎么处理？"),
    "dlg_conflict_delete": ("Delete old files", "删除旧文件"),
    "dlg_conflict_new": ("New timestamped folder", "新建带时间戳的子目录"),
    "dlg_cancel": ("Cancel", "取消"),
    "dlg_error": ("Error", "错误"),
    "err_no_video": ("Please open a video first.", "请先打开视频。"),
    "err_no_position": ("Please set the scanning position.", "请先设置扫描位置。"),
    "err_no_width": ("Please set the strip width.", "请先设置条带宽度。"),
    "err_no_segments": ("Adaptive mode needs segments: click 'Calculate move' or add rows.",
                        "自适应模式需要分段：请点击“计算运动”或手动添加行。"),
    "err_stab_anchor": ("Stabilizer: {msg}", "稳定器：{msg}"),
    "interp_cubic": ("Cubic", "三次 (Cubic)"),
    "interp_lanczos": ("Lanczos", "Lanczos"),
    "interp_linear": ("Linear", "线性 (Linear)"),
    "frame": ("Frame", "帧"),
    "not_set": ("not set", "未设置"),
    "mode_lbl": ("Mode", "模式"),
    "direction": ("Direction", "运动方向"),
    "open_folder": ("Open folder", "打开输出目录"),
    "save_dir_tip": ("Save root directory: {p}", "保存根目录：{p}"),
    "ribbon_tip": ("Live panorama strip — drag horizontally to inspect, double-click to follow newest end",
                   "实时长图滚动带 —— 左右拖拽可回看已拼接部分，双击恢复自动跟随最新端"),
    "ribbon_empty": ("Live panorama strip — stitched result scrolls here in real time during processing",
                     "实时长图滚动带 —— 开始处理后将在此实时滚动显示拼接长图"),
}


def set_language(lang: str) -> None:
    global _LANG
    _LANG = "zh" if lang == "zh" else "en"


def language() -> str:
    return _LANG


def tr(key: str, **kw) -> str:
    en, zh = S.get(key, (key, key))
    s = zh if _LANG == "zh" else en
    return s.format(**kw) if kw else s
