from configparser import Interpolation
from math import ceil
import cv2
import csv 
import numpy as np
import tkinter
from tkinter import filedialog
from skimage.color import rgb2hed, hed2rgb
from openslide import OpenSlide
from skimage.util import img_as_ubyte,img_as_float
from skimage.filters import thresholding, threshold_otsu
### constants

SKIP_CROP = False
MANUAL_THRESHOLD = False
BLOCK_SIZE = 1024

import sys,os

root = tkinter.Tk()
root.withdraw() #use to hide tkinter window

def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename()
    return filename


if len(sys.argv) == 1 :
    FILE_NAME = openfilename()

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
print("###### ARIA-PY v 3.0 ######")
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
            

def resizeAndCrop(src, startX, endX, startY, endY, use_openslide = False) :
    resize_ratio = BLOCK_SIZE/(endX-startX)

    ## first, get total image size
    slide_level = 0
    if use_openslide :
        resized_total_width = src.level_dimensions[0][0] * resize_ratio
        for level in range(src.level_count) :
            if(src.level_dimensions[level][0] < resized_total_width) : break
            slide_level = level
        
        slide_total_width = src.level_dimensions[slide_level][0]

        slide_to_total_ratio = slide_total_width/src.level_dimensions[0][0] 

        slide_startX = int(startX * slide_to_total_ratio)
        slide_startY = int(startY * slide_to_total_ratio)
        slide_endX= int(endX* slide_to_total_ratio)
        slide_endY= int(endY* slide_to_total_ratio)

        slide_to_output_ratio = BLOCK_SIZE / (slide_endX - slide_startX)

        width_height_ratio = (endX - startX) / (endY - startY)
        
        ## np array : height , weight, dim
        result_image = np.zeros((ceil(BLOCK_SIZE * (1/width_height_ratio)),BLOCK_SIZE,3), np.uint8)

        ## Loop through slide to make BLOCK_SIZED pieces
        
        for iterable_x in range(ceil((slide_endX - slide_startX)/BLOCK_SIZE)) :
            for iterable_y in range(ceil((slide_endY - slide_startY)/BLOCK_SIZE)) :

                sys.stdout.write('\033[2K\033[1G')
                print(f"Cropping {(iterable_x * ceil((slide_endY - slide_startY)/BLOCK_SIZE) + iterable_y) / (ceil((slide_endY - slide_startY)/BLOCK_SIZE) * ceil((slide_endX - slide_startX)/BLOCK_SIZE)) * 100}% done", end="\r")

                this_block_width = BLOCK_SIZE

                ## processing when rest of image is less than BLOCK_SIZE (width)
                if(slide_startX + (iterable_x + 1) * BLOCK_SIZE) > slide_endX :
                    this_block_width = slide_endX - (slide_startX + iterable_x * BLOCK_SIZE)

                this_block_height = BLOCK_SIZE
                ## processing when rest of image is less than BLOCK_SIZE (height)
                if(slide_startY + (iterable_y + 1) * BLOCK_SIZE) > slide_endY :
                    this_block_height = slide_endY - (slide_startY + iterable_y * BLOCK_SIZE)
                
                this_block_img = src.read_region((ceil(startX + iterable_x * BLOCK_SIZE / slide_to_total_ratio), ceil(startY + iterable_y * BLOCK_SIZE / slide_to_total_ratio)),slide_level,size = (ceil(this_block_width), ceil(this_block_height)))

                ## Compose destination Xs and Ys
                output_startX = ceil((iterable_x * BLOCK_SIZE)*slide_to_output_ratio)
                output_endX = output_startX + ceil(this_block_width * slide_to_output_ratio)

                output_startY = ceil((iterable_y * BLOCK_SIZE)*slide_to_output_ratio)
                output_endY = output_startY + ceil(this_block_height * slide_to_output_ratio)
                
                ## Compose new image

                result_shape =result_image[output_startY : output_endY, output_startX:output_endX].shape 
                result_image[output_startY : output_endY, output_startX:output_endX] = cv2.cvtColor(cv2.resize(np.array(this_block_img), dsize=(result_shape[1],result_shape[0])), cv2.COLOR_RGB2BGR)

        return result_image

    else :
        ## use opencv
        src = src[newY1:newY2, newX1:newX2]
        if (newX2 - newX1) > BLOCK_SIZE :
            #then resize needed
            contour_resize_ratio = BLOCK_SIZE / (newX2 - newX1)

            result_image = cv2.resize(src, dsize=(BLOCK_SIZE, ceil((newY2 - newY1) * contour_resize_ratio)))
        return result_image
        
        
## end resize function

def deconvolution(src, mask, startX, endX, startY, endY, USE_OPENSLIDE) :
    print("Starting deconvolution...")

    ## get full sized mask image
    resizedMask = cv2.resize(mask, dsize = (endX - startX, endY - startY), interpolation=cv2.INTER_NEAREST)

    ## create empty result image
    result_image = np.zeros((endY - startY,endX - startX), np.uint8) # opencv type, height, weight, dim

    for iterable_x in range(ceil((endX - startX)/BLOCK_SIZE)) :
        for iterable_y in range(ceil((endY - startY)/BLOCK_SIZE)) :

            sys.stdout.write('\033[2K\033[1G')
            print(f"Progress : {((iterable_x * ceil((endY - startY)/BLOCK_SIZE) + iterable_y) / (ceil((endY - startY)/BLOCK_SIZE) * ceil((endX - startX)/BLOCK_SIZE)) * 100):.2f}% done", end="\r")

            this_block_width = BLOCK_SIZE

            ## processing when rest of image is less than BLOCK_SIZE (width)
            if(startX + (iterable_x + 1) * BLOCK_SIZE) > endX :
                this_block_width = endX - (startX + iterable_x * BLOCK_SIZE)

            this_block_height = BLOCK_SIZE
            ## processing when rest of image is less than BLOCK_SIZE (height)
            if(startY + (iterable_y + 1) * BLOCK_SIZE) > endY :
                this_block_height = endY - (startY + iterable_y * BLOCK_SIZE)

            ## define boundaries for full image (output)
            relative_startX = iterable_x * BLOCK_SIZE
            relative_startY = iterable_y * BLOCK_SIZE
            relative_endX = relative_startX + this_block_width
            relative_endY = relative_startY + this_block_height


            ## read defined region from original file 
            if USE_OPENSLIDE :
                this_block_img = np.array(src.read_region((startX + relative_startX, startY + relative_startY),0,size = (this_block_width, this_block_height)))
            else :
                this_block_img = src[(startY+relative_startY) : (startY+relative_startY + this_block_height), (startX + relative_startX) : (startX + relative_startX + this_block_width)]
                this_block_img = cv2.cvtColor(this_block_img,cv2.COLOR_BGR2RGB)

            this_block_img = this_block_img[:,:,:3]

            ## get mask for that region
            this_mask = cv2.cvtColor(resizedMask[relative_startY : relative_endY, relative_startX : relative_endX], cv2.COLOR_GRAY2BGR)

            ## get cropped region, in RGB
            cropped_block = cv2.bitwise_and(this_block_img, this_mask)

            ## clear memory
            del this_block_img

            ## IHC colorspace
            cropped_block = img_as_float(cropped_block)
            cropped_block = rgb2hed(cropped_block)
            null = np.zeros_like(cropped_block[:, :, 0])
            cropped_block = img_as_ubyte(hed2rgb(np.stack((null, null, cropped_block[:, :, 2]), axis=-1)))
            cropped_block = 255 - cv2.cvtColor(cropped_block,cv2.COLOR_RGB2GRAY)
            cropped_block = cv2.bitwise_and(cropped_block, cv2.cvtColor(this_mask,cv2.COLOR_BGR2GRAY))

            ## save to result image
            result_image[relative_startY : relative_endY, relative_startX : relative_endX] = cropped_block

    return result_image
    







## end deconvolution function

USE_OPENSLIDE = False

ix = -1
iy = -1
finalX = -1
finalY = -1
drawing = False

if SKIP_CROP == False :
    try :
        slide_img = OpenSlide(FILE_NAME)
        small_level = slide_img.level_count-1 # if there is less than or equal to 2 levels, set level to last level

        small_img_size = slide_img.level_dimensions[small_level]

        small_img = np.array(slide_img.read_region((0,0), small_level, size=small_img_size)) 


        if(small_img_size[0] > 1024) :
            # then resize to 1024
            small_img_width = 1024
            small_img_height = round(1024*small_img_size[1]/small_img_size[0])
            

            small_img_size = [small_img_width,small_img_height]
            small_img = cv2.resize(small_img, dsize=(small_img_width,small_img_height))
        small_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2BGR)  #finally, the image is presentable with cv2.imshow
        orig_img_size = slide_img.level_dimensions[0]           #get full resolution image size
        USE_OPENSLIDE = True
    except BaseException :
        # try with opencv
        try :
            slide_img = cv2.imread(FILE_NAME)
            # set width to 1024
            small_img_width = 1024
            small_img_height = round(1024*slide_img.shape[0]/slide_img.shape[1])


            small_img_size = [small_img_width,small_img_height]
            small_img = cv2.resize(slide_img, dsize=(small_img_width,small_img_height))
            orig_img_size = [slide_img.shape[1],slide_img.shape[0]]          #get full resolution image size
            USE_OPENSLIDE =False 
        except BaseException :
            print("Error opening image!")
            exit()



    #variables for dragging, initial values

    WINDOW_NAME = "DRAG to set crop area, then press ENTER"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
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

    # del small_img           #will not use again


    resize_ratio = orig_img_size[0]/small_img_size[0]       # get the ratio of larger image compared to thumbnail

    if(ix < finalX) :
        newX1 = int(resize_ratio*ix)                            #start X of new image
        newX2 = int(resize_ratio*finalX)                        #end X of new image
    else :
        newX2 = int(resize_ratio*ix)                            #start X of new image
        newX1 = int(resize_ratio*finalX)                        #end X of new image

    if(iy < finalY) :
        newY1 = int(resize_ratio*iy)                            #start Y of new image
        newY2 = int(resize_ratio*finalY)     #end Y of new image
    else :
        newY2 = int(resize_ratio*iy)                            #start Y of new image
        newY1 = int(resize_ratio*finalY)     #end Y of new image

 
image = resizeAndCrop(slide_img, newX1, newX2, newY1, newY2, USE_OPENSLIDE)

## new X1, X2, Y1, Y2 are coordinates for cropped area using original image
## these will be used later on.

##### now new window - for deconvolution & contour, analysis.
needsReload = True

def passChange(a) :
    global needsReload
    needsReload = True
    pass

WINDOW_NAME2 = "Adjust variables then press ENTER"
cv2.namedWindow(WINDOW_NAME2, cv2.WINDOW_NORMAL)
cv2.createTrackbar("contour start", WINDOW_NAME2, 0, 255, passChange)
cv2.createTrackbar("contour end",WINDOW_NAME2, 0, 255, passChange)
cv2.createTrackbar("min area", WINDOW_NAME2, 1, 100, passChange)
cv2.createTrackbar("background white",WINDOW_NAME2,0,200, passChange)
cv2.createTrackbar("kernel size",WINDOW_NAME2,1,30, passChange)

cv2.setTrackbarPos("contour start",WINDOW_NAME2,10)
cv2.setTrackbarPos("contour end",WINDOW_NAME2,39)
cv2.setTrackbarPos("min area",WINDOW_NAME2,5)
cv2.setTrackbarPos("background white",WINDOW_NAME2,10)
cv2.setTrackbarPos("kernel size",WINDOW_NAME2,3)

origHeight, origWidth, dim = image.shape


## adjust brightness and contrast
BRIGHTNESS = -130
CONTRAST =1.4
imageForCountour = cv2.convertScaleAbs(image, alpha=CONTRAST, beta=BRIGHTNESS)

grayImage = cv2.cvtColor(imageForCountour, cv2.COLOR_BGR2GRAY)
invertedGrayImage = cv2.bitwise_not(grayImage)
height, width= grayImage.shape
totalArea = width*height


keyResult = None
while keyResult != 13 and keyResult != 27:
    if needsReload :
        needsReload = False
        CANNY_START = cv2.getTrackbarPos('contour start', WINDOW_NAME2)
        CANNY_END = cv2.getTrackbarPos('contour end', WINDOW_NAME2)
        AREA_RATIO = cv2.getTrackbarPos('min area', WINDOW_NAME2) / 2000
        if AREA_RATIO == 0:
            AREA_RATIO = 0.001
        WHITE_VALUE = cv2.getTrackbarPos('background white', WINDOW_NAME2)
        KERNEL_SIZE = cv2.getTrackbarPos('kernel size', WINDOW_NAME2)

        canny = cv2.Canny(grayImage, CANNY_START, CANNY_END)
        expanded = cv2.dilate(canny, np.ones([KERNEL_SIZE, KERNEL_SIZE]))
        eroded = cv2.erode(expanded, np.ones([KERNEL_SIZE, KERNEL_SIZE]))
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
        cv2.putText(merged, f"CANNY {CANNY_START} ~ {CANNY_END} / AREA {AREA_RATIO} / WHITE {WHITE_VALUE} / KERNEL {KERNEL_SIZE}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

        cv2.imshow(WINDOW_NAME2, merged)
    keyResult = cv2.waitKey(1)

cv2.destroyAllWindows()

if keyResult == 13:
    # then save
    cv2.imwrite(FILE_NAME + '-area.png', merged)
    cv2.imwrite(FILE_NAME + '-contour.png', finalMaskGray)

else:
    quit()

##
##
##
## RESET (free up memory)
##
##
##
del grayImage
del invertedGrayImage
del expanded
del eroded
del finalMask
del inverted
del merged

from datetime import datetime

## start deconvolution
deconvoluted = deconvolution(slide_img, finalMaskGray, newX1, newX2, newY1, newY2, USE_OPENSLIDE)


cv2.imwrite(FILE_NAME + '-deconvoluted.png',(255-deconvoluted))

###############
## Thresholding
###############


threshold = threshold_otsu(deconvoluted[deconvoluted!=0])

WINDOW_NAME2 = "Adjust threshold then press ENTER"
cv2.namedWindow(WINDOW_NAME2, cv2.WINDOW_NORMAL)
cv2.createTrackbar("threshold", WINDOW_NAME2, 0, 255, passChange)
cv2.setTrackbarPos("threshold",WINDOW_NAME2,threshold)

origHeight, origWidth = deconvoluted.shape
ihc_d_gray_display = deconvoluted.copy()
grayHeight, grayWidth = ihc_d_gray_display.shape
ihc_d_gray_display = cv2.cvtColor(ihc_d_gray_display, cv2.COLOR_GRAY2BGR)
croppedImage = deconvoluted
if origHeight > 1024: ## too large, so resize for display
    croppedImage = cv2.resize(deconvoluted, dsize=(round(1024*origWidth/origHeight),1024))
    ihc_d_gray_display = cv2.resize(ihc_d_gray_display,dsize=(round(1024*grayWidth/grayHeight),1024))


if MANUAL_THRESHOLD :

    keyResult = None
    needsReload =True 
    while keyResult != 13 and keyResult != 27:
        if needsReload :
            needsReload = False
            threshold = cv2.getTrackbarPos('threshold', WINDOW_NAME2)
    
            _,thresholded_image = cv2.threshold(ihc_d_gray_display,threshold,255,cv2.THRESH_BINARY)
            thresholded_image = cv2.cvtColor(thresholded_image,cv2.COLOR_BGR2GRAY)
            cv2.putText(thresholded_image, "threshold : "+str(255 - threshold), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

            cv2.imshow(WINDOW_NAME2, cv2.hconcat([croppedImage,thresholded_image]))
        keyResult = cv2.waitKey(1)

    cv2.destroyAllWindows()
    del thresholded_image


del croppedImage 

_,th_auto = cv2.threshold(deconvoluted,threshold,255,cv2.THRESH_BINARY)
_,th_fixed = cv2.threshold(deconvoluted,55,255,cv2.THRESH_BINARY)
thresholded = cv2.cvtColor(th_auto, cv2.COLOR_GRAY2BGR)
thresholded_fixed = cv2.cvtColor(th_fixed, cv2.COLOR_GRAY2BGR)

cv2.imwrite(FILE_NAME + '-stained(auto).png', 255-thresholded)
cv2.imwrite(FILE_NAME + '-stained(fixed).png', 255-thresholded_fixed)


resizedMask = cv2.resize(finalMaskGray, dsize = (deconvoluted.shape[1], deconvoluted.shape[0]))



fields = ["Total Area", "Threshold", "Stained(auto)", "Stained(fixed)"]
results = [round(np.sum(resizedMask)/255), 255-threshold,round(np.sum(th_auto)/255),round(np.sum(th_fixed)/255) ]
with open(FILE_NAME+"-result.csv", 'w',newline='') as f: 
      
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
      
    write.writerow(fields) 
    write.writerow(results)

print("********************************************")
print(f" Analysis completed for {FILE_NAME}")
print(f" Results saved to {FILE_NAME}-result.csv")
print("********************************************")
