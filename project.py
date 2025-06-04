import os
import math
import logging
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)


class PenguinTextOnImage:
    """PenguinTextOnImage v1.2.0  (baseline & stroke safe)"""

    # ────────── ComfyUI I/O 定义 ───────────
    @classmethod
    def INPUT_TYPES(cls):
        font_dir = os.path.join(os.path.dirname(__file__), "font")
        fonts = [f for f in os.listdir(font_dir)
                 if f.lower().endswith((".ttf", ".ttc", ".otf"))] or ["arial.ttf"]
        return {
            "required": {
                "text":        ("STRING", {"multiline": True}),
                "image":       ("IMAGE",),
                # 位置
                "x_pct":       ("INT", {"default": 50, "min": 0, "max": 100}),
                "y_pct":       ("INT", {"default": 50, "min": 0, "max": 100}),
                "h_anchor":    (["left", "center", "right"],),
                "v_anchor":    (["top", "center", "bottom"],),
                "offset_x":    ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "offset_y":    ("INT", {"default": 0, "min": -4096, "max": 4096}),
                # 字体
                "font_size":   ("INT", {"default": 120, "min": 1, "max": 512}),
                "font_file":   (fonts, {"default": fonts[0]}),
                # 颜色 / 渐变
                "text_color":  ("STRING", {"default": "#ffffff"}),
                "text_opacity":("FLOAT",  {"default": 1.0, "min": 0, "max": 1}),
                "use_gradient":("BOOLEAN",{"default": False}),
                "start_color": ("STRING", {"default": "#ff0000"}),
                "end_color":   ("STRING", {"default": "#0000ff"}),
                "angle":       ("INT", {"default": 0, "min": -180, "max": 180}),
                # 描边
                "stroke_width":("INT", {"default": 8, "min": 0, "max": 128}),
                "stroke_color":("STRING", {"default": "#000000"}),
                "stroke_opacity":("FLOAT", {"default": 1.0, "min": 0, "max": 1}),
                # 阴影
                "shadow_x":    ("INT", {"default": 0, "min": -100, "max": 100}),
                "shadow_y":    ("INT", {"default": 0, "min": -100, "max": 100}),
                "shadow_color":("STRING", {"default": "#000000"}),
                "shadow_opacity":("FLOAT", {"default": 1.0, "min": 0, "max": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_text"
    CATEGORY = "📝 Text/Label"

    # ────────── 小工具 ───────────
    @staticmethod
    def hex_rgba(hexstr, alpha=1.0):
        hexstr = hexstr.lstrip("#")
        if len(hexstr) == 3:
            hexstr = "".join(c*2 for c in hexstr)
        r, g, b = (int(hexstr[i:i+2], 16) for i in (0, 2, 4))
        return (r, g, b, int(max(0, min(1, alpha))*255))

    @staticmethod
    def gradient(w, h, c1, c2, angle=0):
        img = Image.new("RGBA", (w, h))
        rads, proj = math.radians(angle), max(int(abs(w*math.cos(angle))+abs(h*math.sin(angle))), 1)
        cos_a, sin_a = math.cos(rads), math.sin(rads)
        for x in range(w):
            for y in range(h):
                t = (x*cos_a + y*sin_a) / proj
                t = max(0, min(t, 1))
                img.putpixel((x, y), tuple(
                    int(c1[i] + (c2[i]-c1[i])*t) for i in range(3)) + (int(c1[3] + (c2[3]-c1[3])*t),))
        return img

    # ────────── 主逻辑 ───────────
    def apply_text(self, text, image,
                   x_pct, y_pct, h_anchor, v_anchor, offset_x, offset_y,
                   font_size, font_file,
                   text_color, text_opacity,
                   use_gradient, start_color, end_color, angle,
                   stroke_width, stroke_color, stroke_opacity,
                   shadow_x, shadow_y, shadow_color, shadow_opacity):

        if not text.strip():
            return (image,)

        # tensor → PIL
        if isinstance(image, torch.Tensor):
            base = Image.fromarray((image[0].cpu().numpy()*255).astype(np.uint8)).convert("RGBA")
        else:
            base = image.convert("RGBA")
        W, H = base.size

        # 字体
        fpath = os.path.join(os.path.dirname(__file__), "font", font_file)
        try:
            font = ImageFont.truetype(fpath, font_size)
        except:
            font = ImageFont.load_default()

        # ===== 1. 文字 bbox（含行距） =====
        l, t, r, b = font.getbbox(text)          # <-- PIL≥10.0
        bbox_w, bbox_h = r - l, b - t
        ox, oy = -l, -t                           # 把字形整体搬进 (0,0)

        # ===== 2. 图层尺寸：文字 + 描边 =====
        layer_w = bbox_w + stroke_width*2
        layer_h = bbox_h + stroke_width*2
        layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # ===== 3. 阴影 =====
        if (shadow_x or shadow_y) and shadow_opacity > 0:
            draw.text((ox + stroke_width + shadow_x,
                       oy + stroke_width + shadow_y),
                      text, font=font,
                      fill=self.hex_rgba(shadow_color, shadow_opacity))

        # ===== 4. 描边 =====
        if stroke_width > 0 and stroke_opacity > 0:
            sc = self.hex_rgba(stroke_color, stroke_opacity)
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx or dy:
                        draw.text((ox + stroke_width + dx,
                                   oy + stroke_width + dy), text, font=font, fill=sc)

        # ===== 5. 主文字 (单色 / 渐变) =====
        if use_gradient:
            g_img = self.gradient(bbox_w, bbox_h,
                                   self.hex_rgba(start_color, text_opacity),
                                   self.hex_rgba(end_color,   text_opacity),
                                   angle)
            mask = Image.new("L", (bbox_w, bbox_h), 0)
            ImageDraw.Draw(mask).text((0, 0), text, font=font, fill=255)
            layer.paste(g_img, (ox + stroke_width, oy + stroke_width), mask)
        else:
            draw.text((ox + stroke_width, oy + stroke_width),
                      text, font=font,
                      fill=self.hex_rgba(text_color, text_opacity))

        # ===== 6. 计算锚点坐标（仅用纯文字尺寸） =====
        anchor_x = int(W * x_pct / 100) + offset_x
        anchor_y = int(H * y_pct / 100) + offset_y

        if h_anchor == "center":
            dx = anchor_x - bbox_w // 2
        elif h_anchor == "right":
            dx = anchor_x - bbox_w
        else:
            dx = anchor_x  # left

        if v_anchor == "center":
            dy = anchor_y - bbox_h // 2
        elif v_anchor == "bottom":
            dy = anchor_y - bbox_h
        else:
            dy = anchor_y  # top

        # 最终 paste 位置再补回描边偏移
        paste_x = dx - stroke_width
        paste_y = dy - stroke_width

        base.paste(layer, (paste_x, paste_y), layer)

        # 返回 tensor
        out = torch.from_numpy(np.array(base).astype(np.float32) / 255.0)[None,]
        return (out,)
