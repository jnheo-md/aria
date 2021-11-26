import cv2
import numpy
import numpy as np

from skimage.color import rgb2hed, hed2rgb
from openslide import OpenSlide
### constants

SKIP_CROP = False
MANUAL_THRESHOLD = False

import sys
if len(sys.argv) == 1 :
    print("ARIA error : cannot understand arguments")
    print("usage : python aria.py [skip-crop(optional)] [manual-threshold(optional)] [filename]")
    print("ex1) python aria.py skip-crop no-need-to-crop.svs - skips cropping step")
    print("ex1) python aria.py skip-crop manual-threshold no-need-to-crop-manual-threshold.svs - skips cropping step / shows manual threshold adjustment step")
    print("ex2) python aria.py need-to-crop.svs - uses cropping step")
    exit()
elif len(sys.argv) ==2 :
    FILE_NAME = sys.argv[1]
elif len(sys.argv) > 2 :
    if sys.argv[1]  == "skip-crop":
        SKIP_CROP = True
        if sys.argv[2] == "manual-threshold" :
            MANUAL_THRESHOLD= True
            FILE_NAME = sys.argv[3] 
        else :
            FILE_NAME = sys.argv[2] 
    elif sys.argv[1]  == "manual-threshold":
        MANUAL_THRESHOLD= True
        if sys.argv[2] == "skip-crop" :
            MANUAL_THRESHOLD= True
            FILE_NAME = sys.argv[3] 
        else :
            FILE_NAME = sys.argv[2] 
    else : # error
        print("ARIA error : cannot understand arguments")
        print("usage : python aria.py [skip-crop(optional)] [manual-threshold(optional)] [filename]")
        print("ex1) python aria.py skip-crop no-need-to-crop.svs - skips cropping step")
        print("ex1) python aria.py skip-crop manual-threshold no-need-to-crop-manual-threshold.svs - skips cropping step / shows manual threshold adjustment step")
        print("ex2) python aria.py need-to-crop.svs - uses cropping step")
        exit()
    



print("###########################")
print("###### ARIA-PY v 2.2 ######")
print("###########################")
print("#### by JoonNyung Heo  ####")
print("###########################")
print("")
print("Processing file : "+FILE_NAME)
print("###########################")


################################
#  Draws rectangle on the image
################################
def draw_rectangle(x,y):
    global small_img, image 
    image = small_img.copy()    # start from original image
    image -= 50                 # decrease brightness of image

    dragged = np.zeros_like(image)
    cv2.rectangle(dragged, pt1=(ix,iy), pt2=(x, y),color=(255,255,255),thickness=-1)    # create white box
    alpha = 0.8
    mask = dragged.astype(bool)
    image[mask] = cv2.addWeighted(image, alpha, dragged, 1-alpha,0)[mask]               # merge rectangle onto image

def onClick(event, x, y, flags, param):
    global ix, iy, drawing, finalX, finalY
    if event == cv2.EVENT_LBUTTONDOWN:  #when mousedown, set initial values
        # drawing = True
        ix = x
        iy = y

    elif ((event == cv2.EVENT_MOUSEMOVE) & (flags & cv2.EVENT_FLAG_LBUTTON) | (event == cv2.EVENT_LBUTTONUP)):  #when moving, redraw rectangle
        # if drawing == True:
        draw_rectangle(x,y)
        finalX = x
        finalY = y
            


    # elif event == cv2.EVENT_LBUTTONUP:  #when dragging end, set drawing to false and save the final coordinates
    #     # drawing = False
    #     finalX = x
    #     finalY = y
    #     if abs(x-ix)<10 and abs(y-iy)<10 :  # if dragged are is too small, ignore
    #         return
    #     draw_rectangle(x,y)

USE_OPENSLIDE = False

ix = -1
iy = -1
finalX = -1
finalY = -1
drawing = False

if SKIP_CROP == False :
    try :
        slide_img = OpenSlide(FILE_NAME)
        small_level = 5                     #use small image for setting croppable area
        if slide_img.level_count <= 5: small_level = slide_img.level_count-1 # if there is less than or equal to 2 levels, set level to last level

        small_img_size = slide_img.level_dimensions[small_level]

        small_img = np.array(slide_img.read_region((0,0), small_level, size=small_img_size)) 
        small_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2BGR)  #finally, the image is presentable with cv2.imshow
        orig_img_size = slide_img.level_dimensions[0]           #get full resolution image size
        USE_OPENSLIDE = True
    except BaseException :
        # try with opencv
        try :
            slide_img = cv2.imread(FILE_NAME)
            # set width to 1024
            small_img_width = 1024
            small_img_height = round(1024*slide_img.shape[1]/slide_img.shape[0])


            small_img_size = [small_img_height,small_img_width]
            small_img = cv2.resize(slide_img, dsize=(small_img_height,small_img_width))
            orig_img_size = [slide_img.shape[1],slide_img.shape[0]]          #get full resolution image size
            USE_OPENSLIDE =False 
        except BaseException :
            print("Error opening image!")
            exit()



    #variables for dragging, initial values
    


    WINDOW_NAME = "DRAG to set crop area, then press ENTER"
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, onClick)
    image = small_img.copy()

    while True:
        cv2.imshow(WINDOW_NAME, image)
        ret = cv2.waitKey(10)
        if ret == 27:   #esc, exit
            
            exit()
        elif ret==13 :  #return, so continue!
            cv2.destroyAllWindows()
            print("Cropping and exporting image...")
            break
            

    if abs(ix-finalX) < 10 and abs(iy-finalY) < 10 :    # no area or too small area was selected. so export whole image 
        ix = 0
        iy =0
        finalX = small_img_size[0]
        finalY = small_img_size[1]

    del small_img           #will not use again


    resize_ratio = orig_img_size[0]/small_img_size[0]       # get the ratio of larger image compared to thumbnail

    newX1 = int(resize_ratio*ix)                            #start X of new image
    newY1 = int(resize_ratio*iy)                            #start Y of new image
    newX2 = int(resize_ratio*finalX)                        #end X of new image
    newY2 = int(resize_ratio*finalY)     #end Y of new image

    if USE_OPENSLIDE :
        image = np.array(slide_img.read_region((newX1,newY1), 0,size = (newX2-newX1, newY2-newY1)))
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        slide_img.close()
    else :
        image = slide_img[newY1:newY2, newX1:newX2]



else :  ### skip crop

    try :
        slide_img = OpenSlide(FILE_NAME)
        image = np.array(slide_img.read_region((0,0),0,size=slide_img.level_dimensions[0]))
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    except BaseException :
        # try with opencv
        try :
            image = cv2.imread(FILE_NAME)
        except BaseException :
            print("Error opening image!")
            exit()

                   

##### now new window - for deconvolution & contour, analysis.
needsReload = True

def passChange(a) :
    global needsReload
    needsReload = True
    pass

WINDOW_NAME2 = "Adjust variables then press ENTER"
cv2.namedWindow(WINDOW_NAME2)
cv2.createTrackbar("contour start", WINDOW_NAME2, 0, 255, passChange)
cv2.createTrackbar("contour end",WINDOW_NAME2, 0, 255, passChange)
cv2.createTrackbar("min area", WINDOW_NAME2, 1, 100, passChange)
cv2.createTrackbar("background white",WINDOW_NAME2,0,100, passChange)

cv2.setTrackbarPos("contour start",WINDOW_NAME2,10)
cv2.setTrackbarPos("contour end",WINDOW_NAME2,39)
cv2.setTrackbarPos("min area",WINDOW_NAME2,5)
cv2.setTrackbarPos("background white",WINDOW_NAME2,10)


origHeight, origWidth, dim = image.shape
isResized = False

if origWidth > 5000: ## too large, so resize for display
    resized = cv2.resize(image, None, fx=0.3, fy=0.3)
    isResized = True
else:
    resized = image

## adjust brightness and contrast
BRIGHTNESS = -130
CONTRAST =1.4
imageForCountour = cv2.convertScaleAbs(resized, alpha=CONTRAST, beta=BRIGHTNESS)

grayImage = cv2.cvtColor(imageForCountour, cv2.COLOR_BGR2GRAY)
invertedGrayImage = cv2.bitwise_not(grayImage)
# cv2.imshow("original", image)
kernel = np.ones([10,10])
kernel2 = np.ones([10,10])
height, width= grayImage.shape
totalArea = width*height


keyResult = None
while keyResult != 13 and keyResult != 27:
    if needsReload :
        needsReload = False
        CANNY_START = cv2.getTrackbarPos('contour start', WINDOW_NAME2)
        CANNY_END = cv2.getTrackbarPos('contour end', WINDOW_NAME2)
        AREA_RATIO = cv2.getTrackbarPos('min area', WINDOW_NAME2) / 1000
        if AREA_RATIO == 0:
            AREA_RATIO = 0.001
        WHITE_VALUE = cv2.getTrackbarPos('background white', WINDOW_NAME2) / 10

        canny = cv2.Canny(grayImage, CANNY_START, CANNY_END)
        expanded = cv2.dilate(canny, kernel)
        eroded = cv2.erode(expanded, kernel2)
        contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        showingContours = []
        negativeContours = []
        for each in contours:
            area = cv2.contourArea(each)
            if area > AREA_RATIO * totalArea:
                # get average intensitivy in grayscale
                mask = np.zeros_like(invertedGrayImage)  # Create mask where white is what we want, black otherwise
                cv2.drawContours(mask, [each], -1, 255, -1)
                out = np.zeros_like(invertedGrayImage)  # Extract out the object and place into output image
                out[mask == 255] = invertedGrayImage[mask == 255]
                # get sum of grayscale
                average = out.sum() / area
                if average < WHITE_VALUE:  # too white
                    negativeContours.append(each)
                else:
                    showingContours.append(each)

        contourMask = np.zeros_like(invertedGrayImage)
        cv2.drawContours(contourMask, showingContours, -1, 255, -1)
        negativeContourMask = np.zeros_like(invertedGrayImage)
        cv2.drawContours(negativeContourMask, negativeContours, -1, 255, -1)
        finalMaskGray = cv2.bitwise_xor(contourMask, negativeContourMask)
        finalMask = cv2.cvtColor(finalMaskGray, cv2.COLOR_GRAY2BGR)
        finalMask[np.where((finalMask == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  # make it red

        inverted = cv2.cvtColor(invertedGrayImage, cv2.COLOR_GRAY2BGR)
        merged = cv2.addWeighted(inverted, 0.7, finalMask, 0.3, 0)
        cv2.putText(merged, "CANNY "+str(CANNY_START)+" ~ "+str(CANNY_END)+" / AREA "+str(AREA_RATIO)+" / WHITE "+str(WHITE_VALUE), (200,200), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

        # cv2.imshow("Image", cv2.hconcat([imageForCountour, merged]))
        cv2.imshow(WINDOW_NAME2, merged)
    keyResult = cv2.waitKey(1)

cv2.destroyAllWindows()

if keyResult == 13:
    # then save
    # resized = cv2.resize(merged, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(FILE_NAME + '-area.png', merged)

else:
    quit()

##
##
##
## RESET
##
##
##

print("Starting deconvolution process...")

keyResult = None
grayImage = None
invertedGrayImage = None
expanded = None
eroded = None
finalMask = None
inverted = None
merged = None

from datetime import datetime

starttime = datetime.now().timestamp()

# Resize contour to its original size

if isResized:
    finalMaskGray = cv2.resize(finalMaskGray,dsize=(origWidth, origHeight))

colorMask = cv2.cvtColor(finalMaskGray, cv2.COLOR_GRAY2BGR)
croppedImage = cv2.bitwise_and(image, colorMask)

ihc_rgb = cv2.cvtColor(croppedImage,cv2.COLOR_BGR2RGB)
cv2.imwrite(FILE_NAME + '-cropped.png', image)
del image

from skimage.util import img_as_ubyte,img_as_float

# Separate the stains from the IHC image
ihc_rgb = img_as_float(ihc_rgb)
ihc_hed = rgb2hed(ihc_rgb)
del ihc_rgb

null = np.zeros_like(ihc_hed[:, :, 0])
ihc_d_gray = img_as_ubyte(hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))
ihc_d_gray = cv2.cvtColor(ihc_d_gray,cv2.COLOR_RGB2GRAY)
del ihc_hed
print("processing...hed2rgb done, took "+str(datetime.now().timestamp()-starttime))


ihc_d_gray = (255-ihc_d_gray)
ihc_d_gray = cv2.bitwise_and(ihc_d_gray, finalMaskGray)

print("processing... cvtcolor, currently "+str(datetime.now().timestamp()-starttime))

from skimage.filters import threshold_otsu
threshold = threshold_otsu(ihc_d_gray[ihc_d_gray!=0])


####### adjust threshold

WINDOW_NAME2 = "Adjust threshold then press ENTER"
cv2.namedWindow(WINDOW_NAME2)
cv2.createTrackbar("threshold", WINDOW_NAME2, 0, 255, passChange)
cv2.setTrackbarPos("threshold",WINDOW_NAME2,threshold)

origHeight, origWidth, dim = croppedImage.shape
ihc_d_gray_display = ihc_d_gray.copy()
grayHeight, grayWidth = ihc_d_gray_display.shape
ihc_d_gray_display = cv2.cvtColor(ihc_d_gray_display, cv2.COLOR_GRAY2BGR)
if origHeight > 1024: ## too large, so resize for display
    croppedImage = cv2.resize(croppedImage, dsize=(round(1024*origWidth/origHeight),1024))
    ihc_d_gray_display = cv2.resize(ihc_d_gray_display,dsize=(round(1024*grayWidth/grayHeight),1024))



## adjust brightness and contrast
BRIGHTNESS = -130
CONTRAST =1.4
imageForCountour = cv2.convertScaleAbs(resized, alpha=CONTRAST, beta=BRIGHTNESS)

grayImage = cv2.cvtColor(imageForCountour, cv2.COLOR_BGR2GRAY)
invertedGrayImage = cv2.bitwise_not(grayImage)
kernel = np.ones([10,10])
kernel2 = np.ones([10,10])
height, width= grayImage.shape
totalArea = width*height

if MANUAL_THRESHOLD :

    keyResult = None
    needsReload =True 
    while keyResult != 13 and keyResult != 27:
        if needsReload :
            needsReload = False
            threshold = cv2.getTrackbarPos('threshold', WINDOW_NAME2)
    
            _,thresholded_image = cv2.threshold(ihc_d_gray_display,threshold,255,cv2.THRESH_BINARY)
            cv2.putText(thresholded_image, "threshold : "+str(threshold), (200,200), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

            cv2.imshow(WINDOW_NAME2, cv2.hconcat([croppedImage,thresholded_image]))
        keyResult = cv2.waitKey(1)

    cv2.destroyAllWindows()
    del thresholded_image


del croppedImage 


_,th_auto = cv2.threshold(ihc_d_gray,threshold,255,cv2.THRESH_BINARY)
_,th_fixed = cv2.threshold(ihc_d_gray,55,255,cv2.THRESH_BINARY)

del ihc_d_gray
print("processing... thresholded done,  currently "+str(datetime.now().timestamp()-starttime))
thresholded = cv2.cvtColor(th_auto, cv2.COLOR_GRAY2BGR)
thresholded_fixed = cv2.cvtColor(th_fixed, cv2.COLOR_GRAY2BGR)

print("processing... cvtcolor2 done, currently "+str(datetime.now().timestamp()-starttime))

cv2.imwrite(FILE_NAME + '-stained(auto).png', 255-thresholded)
cv2.imwrite(FILE_NAME + '-stained(fixed).png', 255-thresholded_fixed)
print("processing... imgwrite done, currently "+str(datetime.now().timestamp()-starttime))

import csv 

fields = ["Total Area", "Threshold", "Stained(auto)", "Stained(fixed)"]
results = [round(numpy.sum(finalMaskGray)/255), 255-threshold,round(numpy.sum(th_auto)/255),round(numpy.sum(th_fixed)/255) ]
with open(FILE_NAME+"-result.csv", 'w',newline='') as f: 
      
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
      
    write.writerow(fields) 
    write.writerow(results)
