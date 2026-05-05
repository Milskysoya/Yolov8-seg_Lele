# gesture_drawer.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class GestureDrawer:
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        self.custom_images = {}
        self.use_custom = False

        # Coba load font, fallback ke default jika tidak ada
        try:
            # Windows
            self.font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 36)
        except Exception:
            try:
                # Linux
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except Exception:
                self.font = ImageFont.load_default()

    def load_custom_images(self, image_paths_dict):
        try:
            for finger_count, path in image_paths_dict.items():
                img = cv2.imread(path)
                if img is not None:
                    img_resized = cv2.resize(img, (self.width, self.height))
                    self.custom_images[finger_count] = img_resized
                    print(f"Gambar {finger_count} jari loaded: {path}")
                else:
                    print(f"Gagal load gambar {finger_count} jari: {path}")
            if len(self.custom_images) > 0:
                self.use_custom = True
        except Exception as e:
            print(f"Error loading custom images: {e}")

    def create_gesture_image(self, finger_count):
        if self.use_custom and finger_count in self.custom_images:
            return self.custom_images[finger_count]

        img = Image.new('RGB', (self.width, self.height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        draw_map = {
            0: self._draw_fist,
            1: self._draw_one_finger,
            2: self._draw_two_fingers,
            3: self._draw_three_fingers,
            4: self._draw_four_fingers,
            5: self._draw_five_fingers,
        }

        if finger_count in draw_map:
            draw_map[finger_count](draw)
        else:
            self._draw_many_fingers(draw, finger_count)

        img_array = np.array(img)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    def _draw_text_center(self, draw, text, color=(255, 255, 255)):
        # Hitung posisi tengah teks
        bbox = draw.textbbox((0, 0), text, font=self.font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (self.width - tw) // 2
        y = (self.height - th) // 2
        draw.text((x, y), text, fill=color, font=self.font)

    def _draw_fist(self, draw):
        draw.ellipse([50, 50, 350, 350], fill=(100, 100, 100), outline=(0, 0, 0), width=3)
        self._draw_text_center(draw, "KEPALAN")

    def _draw_one_finger(self, draw):
        draw.ellipse([75, 75, 325, 325], fill=(255, 0, 0), outline=(0, 0, 0), width=3)
        self._draw_text_center(draw, "1 JARI")

    def _draw_two_fingers(self, draw):
        points = [(200, 50), (350, 300), (50, 300)]
        draw.polygon(points, fill=(0, 200, 0), outline=(0, 0, 0), width=3)
        self._draw_text_center(draw, "2 JARI", color=(0, 0, 0))

    def _draw_three_fingers(self, draw):
        points = [(200, 50), (350, 320), (50, 320)]
        draw.polygon(points, fill=(255, 200, 0), outline=(0, 0, 0), width=3)
        self._draw_text_center(draw, "3 JARI", color=(0, 0, 0))

    def _draw_four_fingers(self, draw):
        draw.rectangle([50, 50, 350, 350], fill=(0, 0, 220), outline=(0, 0, 0), width=3)
        self._draw_text_center(draw, "4 JARI")

    def _draw_five_fingers(self, draw):
        center = (200, 200)
        radius = 130
        points = []
        for i in range(5):
            angle = i * 2 * np.pi / 5 - np.pi / 2
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=(220, 0, 220), outline=(0, 0, 0), width=3)
        self._draw_text_center(draw, "5 JARI")

    def _draw_many_fingers(self, draw, finger_count):
        draw.rectangle([50, 50, 350, 350], fill=(100, 0, 100), outline=(0, 0, 0), width=3)
        self._draw_text_center(draw, f"{finger_count} JARI")
