import base64
import sys

with open('images/'+sys.argv[1], "rb") as img_file:
    my_string = base64.b64encode(img_file.read())
print(my_string.decode("utf-8"))
