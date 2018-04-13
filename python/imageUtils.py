import numpy
import cv2
from PIL import Image
import imagehash
import colorsys
import scipy
import scipy.misc
import scipy.cluster
# import Configuration as cfg
import os
import PIL.ImageOps
import datetime, time
import scipy.spatial

DEBUG = True
OUTPUTDIRECTORY = 'Output'
if not os.path.exists(OUTPUTDIRECTORY):
    os.makedirs(OUTPUTDIRECTORY)

def get_timestamp():
    return str(datetime.datetime.fromtimestamp(int('{:.0f}'.format(time.time())))).replace(' ', '_').replace(':', '')

def cvToMatplot(cvImage):
    """

    Args:
        cvImage:

    Returns:

    """
    if type(cvImage) == PIL.Image.Image:
        cvImage = pilToCv(cvImage)
    # print(cvImage.shape)
    if len(cvImage.shape) > 2:
        return cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(cvImage, cv2.COLOR_GRAY2RGB)


def getVBlock(image, maskFrom, maskTo):
    """ Get only the part of the image specified by maskFrom and maskTo X-axis

    Args:
        image: Image to be masked
        maskFrom: starting X axis point of block
        maskTo: ending X axis point of block

    Returns:
        Block of the image
    """
    maskImage = numpy.zeros(image.shape, dtype=numpy.uint8)
    maskImage[maskFrom:maskTo, :] = image[maskFrom:maskTo, :]
    return maskImage


def getHBlock(image, maskFrom, maskTo):
    """ Get only the part of the image specified by maskFrom and maskTo X-axis

    Args:
        image: Image to be masked
        maskFrom: starting X axis point of block
        maskTo: ending X axis point of block

    Returns:
        Block of the image
    """
    maskImage = numpy.zeros(image.shape, dtype=numpy.uint8)
    maskImage[:, maskFrom:maskTo] = image[:, maskFrom:maskTo]
    return maskImage

def getRBlock(image, maskFrom, maskTo):
    img = image.copy()
    maskImage = numpy.zeros(image.shape, dtype=numpy.uint8)
    yPoint1 = 0
    yPoint2 = int(img.shape[0] * 0.30)
    maskImage[yPoint1:yPoint2, maskFrom:maskTo] = img[yPoint1:yPoint2, maskFrom:maskTo]
    return maskImage

def getExtremePoints(image):
    """Get extreme corner points from top bottom left right.

    Args:
        image: image or the block

    Returns:
        tuple of extreme points as (bottom, left, top, right)
    """
    img = image.copy()
    dImage = image.copy()
    try:
        c = getLargestContour(img.copy())
    except Exception as e:
        print(e)
        print('Blank Image')
        cv2.imshow('Exception', img)
        cv2.waitKey(0)
        return None, None, None, None
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    if DEBUG: # or cfg.DEBUG:
        # if True:
        print(dImage.shape)
        cv2.drawContours(dImage, [c], -1, (0, 255, 255), 2)
        cv2.circle(dImage, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(dImage, extRight, 8, (0, 255, 0), -1)
        cv2.circle(dImage, extTop, 8, (255, 0, 0), -1)
        cv2.circle(dImage, extBot, 8, (255, 255, 0), -1)

        # show the output image
        # cv2.imshow("Image", dImage)
        # cv2.waitKey(0)
    return (extBot, extLeft, extTop, extRight)

def getpHash(image):
    """Get pHash of an Image of PIL object type
    
    Args:
        image: {PIL.PngImagePlugin.PngImageFile} -- Input Image

    Returns:
        Hash Object with PHash
    """
    if type(image) == numpy.ndarray:
        image = cvToPil(image)
    imgPHash = imagehash.phash(image)
    return imgPHash


def getwhash(image):
    """

    Args:
        image:

    Returns:

    """
    if type(image) == numpy.ndarray:
        image = cvToPil(image)
    whash = imagehash.whash(image)
    # print("whash:", whash)
    return whash


def getaHash(image):
    """Get pHash of an Image of PIL object type

    Args:
        image: Input Image

    Returns:
        Hash Object with Average Hash

    """
    if type(image) == numpy.ndarray:
        image = cvToPil(image)
    ahash = imagehash.average_hash(image)
    # print("ahash:", ahash)
    return ahash


def cvToPil(cvImage):
    """Converts opencv format image to PIL object
    
    Args:
        cvImage: {numpy.ndarray} -- OpenCV or ndarray Image

    Returns:
        Image converted to PIL Image

    """
    pilImage = Image.fromarray(cvImage)
    return pilImage


def pilToCv(pilImage):
    """Converts PIL Image to ndarray (opencv Image)
    
    Args:
        pilImage: {PIL.PngImagePlugin.PngImageFile} -- Image of PIL Format

    Returns:
        Image converted to numpy.ndarray (OpenCV supported Format)

    """
    openCvImage = numpy.array(pilImage)
    # open_cv_image = openCvImage[:, :, ::-1].copy()
    return openCvImage


def getLargestContour(cvImage, debug=False):
    """Returns the largest Countour from an Image
    
    Args:
        cvImage: {numpy.array} -- Input CV Image (BGR or Gray)
    
    Returns:
        numpy.array -- Largest Countour
    """
    img = cvImage.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if debug or DEBUG:
        cv2.imshow('getLargestContour', img)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    cnt = max(contour_sizes, key=lambda x: x[0])[1]
    return cnt

def resizeWithAspect(image,MAX_SIZE = 375):
    # file = os.path.split(image_path)[1]
    # print(file)
    reconvert = False
    if type(image) == numpy.ndarray:
        image = cvToPil(image)
        reconvert = True

    # image = Image.open(image_path)
    # original_size = max(image.size[0], image.size[1])

    resized_width = MAX_SIZE
    resized_height = getNewHeight(image, resized_width)

    pilImage = image.resize((resized_width, resized_height), Image.ANTIALIAS)
    # image.show()
    # image.save(file[:-4] + '__.png')
    if reconvert:
        pilImage = pilToCv(pilImage)
        
    return pilImage


def resizedWithHeight(image, MAX_SIZE=375):
    # file = os.path.split(image_path)[1]
    # print(file)
    reconvert = False
    if type(image) == numpy.ndarray:
        image = cvToPil(image)
        reconvert = True

    # image = Image.open(image_path)
    # original_size = max(image.size[0], image.size[1])

    resized_height= int(MAX_SIZE)
    resized_width = getNewWidth(image, resized_height)

    pilImage = image.resize((resized_width, resized_height), Image.ANTIALIAS)
    # image.show()
    # image.save(file[:-4] + '__.png')
    if reconvert:
        pilImage = pilToCv(pilImage)

    return pilImage

def resizeIfReq(image):
    """Resizes the image by with factor 0.5 until its width is less than 720

    Args:
        image: {numpy.ndarray} -- Input Image

    Returns:
        Resized Image

    """
    while image.shape[0] > 720 or image.shape[1] > 720:
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    return image


def cropImage(image, coords, padding=False):
    """ For Cropping an image

    Args:
        image: Input Image
        coords: list of coordinates top-left and bottom-right
        padding: Boolean, if want to add padding to the cropped Image

    Returns:
        cropped Image

    """
    img = image.copy()
    # img = img[]
    print(coords)
    if len(coords) == 2:
        point1 = coords[0]
        point2 = coords[1]
        img = img[point1[1]:point2[1], point1[0]:point2[0]]
    else:
        maxX, minX = max(coords[:, :1])[0], min(coords[:, :1])[0]
        maxY, minY = max(coords[:, 1:])[0], min(coords[:, 1:])[0]
        print(maxX, maxY)
        print(minX, minY)
        img = img[minY:maxY, minX:maxX]
    if padding:
        newImg = numpy.zeros((img.shape[0] + 20, img.shape[1] + 20), dtype=numpy.uint8)
        newImg[10:10 + img.shape[0], 10:10 + img.shape[1]] = img
    else:

        newImg = numpy.zeros((img.shape[0], img.shape[1]), dtype=numpy.uint8)
        newImg[:img.shape[0], :img.shape[1]] = img


        return img

    cv2.imshow('cropped', newImg)
    cv2.imshow('OG', img)
    cv2.imwrite('images/testCropped.png', newImg)

    # cv2.waitKey(0)
    return newImg


def replaceWithTransparency(img, replace=[255, 255, 255]):
    """ Creates an alpha channel and opaques the color passed with replace parameter.
    Default is White Color

    Args:
        img: Image on which the alpha channel is to be added
        replace: The color which needs to be made opaque

    Returns:
        image: with alpha channel and opaqued color.
    """
    newImage = img.convert('RGBA')
    imgData = newImage.getdata()
    newData = []
    for pixelData in imgData:
        if pixelData[0] == replace[0] and pixelData[1] == replace[1] and pixelData[2] == replace[2]:
            newData.append((replace[0], replace[1], replace[2], 0))
        else:
            newData.append(pixelData)
    newImage.putdata(newData)
    return newImage


def rotateImage(mat, angle):
    """

    Args:
        mat:
        angle:

    Returns:

    """
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def getDistance(point1, point2):
    """ Returns Distance between two points

    Args:
        point1: Point 1
        point2: Point 2

    Returns:
        Returns Distance between two points
    """
    return numpy.sqrt(numpy.square(point1[0] - point2[0]) + numpy.square(point1[1] - point2[1]))


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def getObjectCoords(cvImage, minArea=False):
    """ To get Bounding Box coordinates of Largest Contour in an image.

    Args:
        cvImage: Image having the shape
        minArea: True if coordinates with minimum area is required.
                 False will use bounding Rectangle.

    Returns:
        Clockwise coordinates starting from Bottom-left

    """
    contour = getLargestContour(cvImage)
    if minArea:
        rect = cv2.minAreaRect(contour)
        print(rect)
        myContourAngle = rect[2]
        if (rect[0][0]-rect[1][0]) < (rect[0][1]-rect[1][1]):
            myContourAngle -= 90
        rotImage = rotateImage(cvImage.copy(), myContourAngle)
        contour = getLargestContour(rotImage)
        # box = cv2.boxPoints(rect)
        # box = numpy.int0(box)
        # print(box)
        # coords = list(map(tuple, box))
        # return coords[0], coords[1], coords[2], coords[3]
    # else:
    coords = cv2.boundingRect(contour)
    return coords[0], coords[1], coords[2], coords[3]


def addPadding(cvImage):
    """ Adds Padding to the image

    Args:
        cvImage: Input Image

    Returns:
        Image with padding of 10px from every side.

    """
    reconvert = False
    if type(cvImage) != numpy.ndarray:
        img = pilToCv(cvImage)
        # image = cvToPil(image)
        reconvert = True
    else:
        img = cvImage.copy()
    # img = cvImage
    if len(img.shape) > 2:
        newImage = numpy.ones((img.shape[0] + 20, img.shape[1] + 20, 3), dtype=numpy.uint8)
    else:
        newImage = numpy.zeros((img.shape[0] + 20, img.shape[1] + 20), dtype=numpy.uint8)
    newImage = newImage*255
    newImage[10:10 + img.shape[0], 10:10 + img.shape[1]] = img
    cv2.imwrite(os.path.join(OUTPUTDIRECTORY, "padding.png"),newImage) # cfg.IMAGEDUMP instead of OUTPUTDIRECTORY
    if reconvert:
        newImage = cvToPil(newImage)
    return newImage



def getArea(cvImage):
    """ Returns area of the largest Contour
    Args:
        image: cvImage {numpy.ndarray} -- OpenCV or ndarray Image

    Returns:
        return the area of cvImage
    """
    image = cvImage.copy()
    # grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cnt = getLargestContour(image)
    area = cv2.contourArea(cnt)
    return area

def invertPilImage(image):
    if type(image) == numpy.ndarray:
        image = cvToPil(image)

    # image = Image.open('your_image.png')

    inverted_image = PIL.ImageOps.invert(image)

    # inverted_image.save('new_name.png')
    return inverted_image


def resizeImage(image , w , h ):
    """
    Args:
        pilImage: Convert PIL image to fixed sized of 375*101

    Returns:
        return resize PIL image
    """
    reconvert = False
    if type(image) == numpy.ndarray:
        image = cvToPil(image)
        reconvert = True
    new_width =  cfg.SHAPE_WIDTH
    new_height =  cfg.SHAPE_HEIGHT
    print(w, h)
    pilImage = image.resize((int(w), int(h)), Image.ANTIALIAS)
    if reconvert:
        pilImage = pilToCv(pilImage)
    # new_height = new_width / image.size[0] * image.size[1]
    # pilImage = image.resize((new_width, new_height), Image.ANTIALIAS)
    return pilImage




def rgbToHsv(r, g, b):
    """
    Args:
        r: pass the r value
        g: pass the g value
        b: pass the b value

    Returns:
        return the hsv value of CvImage
    """
    # r_clr = r/255
    # g_clr = g/255
    # b_clr = b/255
    # h,s,v = colorsys.rgb_to_hsv(r_clr,g_clr,b_clr)
    # hsv_value =360 *h, 100 * s, 100 * v
    img = numpy.zeros((1, 1, 3), numpy.uint8)
    img[:] = [b, g, r]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = img_hsv[0][0][0], img_hsv[0][0][1], img_hsv[0][0][2]
    return h, s, v


def getColor(cvImage, roi=None):
    """
    Args:
        cvImage: pass the cvImage to get the RBG color of image

    Returns:
        return the RGB value of cvImage
    """
    NUM_CLUSTERS = 1
    # thresh = 75
    image = cvImage.copy()
    # image = cv2.GaussianBlur(cvImage.copy(), (5, 5), 0)
    if roi is None:
        h, w, _ = cvImage.shape
        distance = getDistance((0, 0), (w, h))
        # print('DistanceX:', distanceX)
        # distanceY = getDistance((0,0), (0,h))
        # print('DistanceY:', distanceY)
        # threshX = int(distanceX * 0.05)
        thresh = int(distance * 0.01)
        # print('threshX:', threshX)

        # print('threshY:', threshY)
        x1, x2, y1, y2 = int(w / 2) - thresh, int(w / 2) + thresh, int(h / 2) - thresh, int(h / 2) + thresh
        roi = cvImage[y1:y2, x1:x2]
        roied = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow('GetColor ROI', roied)
    else:
        roi = image

    # cv2.waitKey(0)
    # print(cvImage.shape)
    RGB_Image = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2RGB)
    ar = numpy.asarray(RGB_Image)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    rgb_value, _ = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    # cv2.imwrite(os.path.join(OUTPUTDIRECTORY, "clusterImage.png"),RGB_Image) # cfg.IMAGEDUMP instead of OUTPUTDIRECTORY
    rgb_value = numpy.int8(numpy.ceil(rgb_value))
    r, g, b = rgb_value[0][0], rgb_value[0][1], rgb_value[0][2]
    colImage = numpy.zeros((20, 20, 3), dtype=numpy.uint8)
    colImage[:, :, :] = b, g, r
    # cv2.imshow('Found Color', colImage)
    return r, g, b


class CvImage:
    """
    """

    def __init__(self, image_path, flag=1):
        """ Creates cvImage object, to manage data related to the image in the object

        Args:
            image_path: Path of the image to open
            flag: 1 for color, 0 for grayscale
        """
        self.image = cv2.imread(image_path, flag)
        self.OGimage = self.image.copy()
        # self.botLeft, self.topLeft, self.topRight, self.botRight = None # getObjectCoords(self.image)
        # self.diagonalDistance = getDistance(self.topLeft, self.botRight)
        self.area = None
        self.phash = None
        # self.blockLeft = Block((0,0), (0,0))
        # self.blockTop = Block((0,0), (0,0))
        # self.blockRight = Block((0,0), (0,0))
        # self.blockBottom = Block((0,0), (0,0))



def flipHorizontallyIfReq(image):
    """Flips the image horizontally if the number of white pixels is more on the left side of shape

    Args:
        image: Image having the shape

    Returns:
        flipped Image
    """
    # img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    img = image.copy()
    (_, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    tempImage = image.copy()
    contour = getLargestContour(img)
    cv2.drawContours(tempImage, [contour], -1, (127, 255, 127))
    x, _, w, _ = cv2.boundingRect(contour)
    midX = int(x + (w / 2))
    # tempImage = cv2.circle(tempImage, (midX, int(y + (h / 2))), 3, (255, 0, 0), thickness=3)
    # cv2.imshow('MidX', tempImage)
    imgLeft = img[:, :midX+1]
    imgRight = img[:, midX:]
    # print(imgLeft.shape)
    if DEBUG: # or cfg.DEBUG:
        cv2.imshow('HorizontalLeft', imgLeft)
        cv2.imshow('HorizontalRight', imgRight)
    # totLeft = imgLeft.shape[0] * imgLeft.shape[1]
    # totRight = imgRight.shape[0] * imgRight.shape[1]
    countLeft = cv2.countNonZero(imgLeft)
    countRight = cv2.countNonZero(imgRight)
    print('Horizontal Count')
    print('Left: ', countLeft, 'Right: ', countRight)
    if countLeft > countRight:
        img = cv2.flip(image.copy(), 1)
        return img
    # cv2.imshow('Rot', image)
    return image


def flipVerticallyIfReq(image):
    """Flips the image vertically if the number of white pixels is more on the Top side of shape

    Args:
        image: Image having the shape

    Returns:
        flipped Image
    """
    # img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    img = image.copy()
    tempImage = image.copy()
    (_, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # contour = getLargestContour(img)
    # cv2.drawContours(tempImage, [contour], -1, (127, 255, 127))
    # x, y, w, h = cv2.boundingRect(contour)
    x, y, w, h = getObjectCoords(img.copy(), minArea=False)
    print(x, y, w, h)
    midY = int(y + (h / 2))
    midX = int(x + (w / 2))
    print('midY', midY)
    print('midX', midX)
    # tempImage = cv2.circle(cv2.cvtColor(tempImage, cv2.COLOR_GRAY2BGR), (midX, midY), 5, (255, 0, 0))
    # cv2.imshow('MidY', tempImage)
    imgTop = img.copy()[:midY+1, :]
    imgBottom = img.copy()[midY:, :]
    print(imgBottom.shape)
    if DEBUG: # or cfg.DEBUG:
        cv2.imshow('Top', imgTop)
        cv2.imshow('Bottom', imgBottom)

    # print(imgTop.shape)
    # totTop = imgTop.shape[0] * imgTop.shape[1]
    # totBottom = imgBottom.shape[0] * imgBottom.shape[1]
    countTop = cv2.countNonZero(imgTop)
    countBottom = cv2.countNonZero(imgBottom)
    print('Vertical Count')
    print('Top: ', countTop, 'Bottom:', countBottom)
    if countTop > countBottom:
        if DEBUG: # or cfg.DEBUG:
            cv2.imwrite(os.path.join(OUTPUTDIRECTORY, 'beforeVFlip.png'), tempImage) # cfg.IMAGEDUMP instead of OUTPUTDIRECTORY
        img = cv2.flip(image.copy(), 0)
        if DEBUG: # or cfg.DEBUG:
            cv2.imwrite(os.path.join(OUTPUTDIRECTORY, 'AfterVFlip.png'), img) # cfg.IMAGEDUMP instead of OUTPUTDIRECTORY
        return img
    # cv2.imshow('V - FLip', img)
    return image

def getNewHeight(inputImage, newWidth = 375):
    if not type(inputImage) == numpy.ndarray:
        image = pilToCv(inputImage)
    else:
        image = inputImage
    h, w = image.shape[:2]
    aspectRatio = w / h
    newHeight = int(numpy.ceil(newWidth / aspectRatio))
    return newHeight

def getNewWidth(inputImage, newHeight = 375):
    if not type(inputImage) == numpy.ndarray:
        image = pilToCv(inputImage)
    else:
        image = inputImage
    h, w = image.shape[:2]
    aspectRatio = w / h
    newWidth = int(numpy.ceil(newHeight / aspectRatio))
    return newWidth


def getPercentSimilarity(inputImage, dbImage):
    # try:
    # print(type(inputImage), type(dbImage))
    invertedInputImage = (255 - inputImage)
    invertedDBImage = (255 - dbImage)
    inputPhash = getpHash(invertedInputImage)
    dbPhash = getpHash(invertedDBImage)
    phash_diff = (inputPhash - dbPhash) / len(inputPhash.hash) ** 2
    per = 100 - (100 * phash_diff)
    # newInputImageHeight = getNewHeight(invertedInputImage, newWidth=invertedDBImage.shape[1])
    # print('invertedInputImage.shape', invertedInputImage.shape)
    # invertedInputImage = cv2.resize(invertedInputImage, (invertedDBImage.shape[1], newInputImageHeight))
    # print('invertedInputImage.shape', invertedInputImage.shape)
    # # per = 0
    # cv2.imshow('ProcessedImage', invertedInputImage)
    # cv2.imshow('DBImage', invertedDBImage)
    # # cv2.waitKey(0)
    # if invertedInputImage.shape[0] <= invertedDBImage.shape[0]:
    #     heightDiff = invertedDBImage.shape[0] - invertedInputImage.shape[0]
    #     blackBgEmptyImage = numpy.zeros((invertedDBImage.shape[0], invertedDBImage.shape[1]), numpy.uint8)
    #     blackBgEmptyImage[heightDiff:, :] = invertedInputImage
    #     cv2.imshow('ProcessedImageB', blackBgEmptyImage)
    #     cv2.imshow('DBImageB', invertedDBImage)
    #     per = scipy.spatial.distance.euclidean(numpy.ravel(blackBgEmptyImage), numpy.ravel(invertedDBImage))
    #     print('per before calc', per)
    #     per = 100 - 100 * (per / len(numpy.ravel(blackBgEmptyImage)))
    #     print('per', per)
    #     # diff = cv2.bitwise_xor(temp, invertedDBImage)
    #     # diffCount = cv2.countNonZero(diff)
    #     # countDB = cv2.countNonZero(invertedDBImage)
    #     # per = 100 - (diffCount / countDB * 100)
    # else:
    #     heightDiff = invertedInputImage.shape[0] - invertedDBImage.shape[0]
    #     blackBgEmptyImage = numpy.zeros((invertedInputImage.shape[0], invertedInputImage.shape[1]), numpy.uint8)
    #     blackBgEmptyImage[heightDiff:, :] = invertedDBImage
    #     cv2.imshow('ProcessedImageB', invertedInputImage)
    #     cv2.imshow('DBImageB', blackBgEmptyImage)
    #     per = scipy.spatial.distance.euclidean(numpy.ravel(invertedInputImage), numpy.ravel(blackBgEmptyImage))
    #     print('per before calc', per)
    #     per = 100 - 100 * (per / len(numpy.ravel(invertedInputImage)))
    #     print('per', per)
    # #     per = scipy.spatial.distance.euclidean(numpy.ravel(temp), numpy.ravel(invertedInputImage))
    # #     # diff = cv2.bitwise_xor(temp, invertedInputImage)
    # #     # diffCount = cv2.countNonZero(diff)
    # #     # countDB = cv2.countNonZero(invertedInputImage)
    # #     # per = 100 - (diffCount / countDB * 100)
    # #     per = 100 * (per / len(numpy.ravel(temp)))
    # # print('Percent:', per)
    # print('heightDiff', heightDiff)
    # cv2.waitKey(0)
    return per
    # except Exception as e:
    #     print(e)


if __name__ == "__main__":
    # r =141
    # g=143
    # b=128
    # rbg_to_hsv(r,g,b)
    image = cv2.imread(r"D:\test\DR35_.png")
    addPadding(image)
