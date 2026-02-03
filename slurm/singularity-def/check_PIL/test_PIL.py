from PIL import Image, ImageDraw, ImageFont

# 創建一個空白圖像
image = Image.new('RGB', (200, 100), color = (255, 255, 255))
draw = ImageDraw.Draw(image)

# 加載字體
font_path = "PIL.ttf"
font = ImageFont.truetype(font_path, 20)

# 測試 getsize 方法
text = "Hello, World!"
try:
    w, h = font.getsize(text)
    print(f"Text width: {w}, Text height: {h}")
except AttributeError as e:
    print(f"Error: {e}")

# 在圖像上繪製文本
draw.text((10, 10), text, font=font, fill=(0, 0, 0))

# 保存圖像以驗證文本渲染
image.save("PIL_test_image.png")

