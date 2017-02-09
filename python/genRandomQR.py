# description = 'For Generating URL QR code(s) appending random 6 digit number to the url.'

import sys
from qrcode import *  # pip install qrcode pillow
from random import randrange
from os import makedirs
from os.path import exists

if not len(sys.argv) == 3:
    print("USAGE: genRandomQR.py [Number of QRCodes] [Directory]")
    exit(0)
qr = QRCode(version=4, error_correction=ERROR_CORRECT_L, border=2)
numOfPics = int(sys.argv[1])
directory = sys.argv[2]
if not exists(directory):
    makedirs(directory)
while numOfPics:
    randnumStr = str(randrange(100000, 1000000, 6))
    urlqr = "http://www.example.com/" + randnumStr
    qr.add_data(urlqr)
    qr.make(fit=True)
    img = qr.make_image()
    imgfile = directory + randnumStr + ".png"
    img.save(imgfile)
    qr.clear()
    numOfPics -= 1
