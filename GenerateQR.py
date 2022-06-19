import pyqrcode
import png
from pyqrcode import QRCode

app_link = "https://share.streamlit.io/ovi2701/licenta_maskpredictionapp/main/MaskApp.py"

create_qr = pyqrcode.create(app_link)

create_qr.png("MaskAppQR.png", scale=6)