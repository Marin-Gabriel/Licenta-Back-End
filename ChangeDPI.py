from PIL import Image

im = Image.open('images/3.jpg')
im.save("DPI-Changed/test-300.jpg", dpi=(300,300))
