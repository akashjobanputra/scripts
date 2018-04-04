gimpHSV = [70, 10, 83]
cvHSV = []
cvHSV.append(int(gimpHSV[0] / 2))
cvHSV.append(int((gimpHSV[1] / 100) * 255))
cvHSV.append(int((gimpHSV[2] / 100) * 255))
print(cvHSV)
