import numpy as np
import math
from matplotlib import pyplot as plt


def get_borders(image):
    '''
    image : pixel array
    return 4-element tuple where each element specifies the first row/column in which a non-zero value appears from each side
    '''
    m = len(image)
    n = len(image[0])
    top = m
    bottom = 0
    left = n
    right = 0
    for i in range(m):
        for j in range(n):
            if image[i][j] > 0:
                top = min(i,top)
                bottom = max(i,bottom)
                left = min(j,left)
                right = max(j,right)
    return (top,left,right,bottom)

def fit_to_box(image):
    '''
    image : pixel array

    new_image : a 20x20 pixel version of the original image
    
    '''

    #TODO make box_size a parameter
    BOX_SIZE = 20

    top,left,right,bottom = get_borders(image)
    if bottom - top > right - left:
        left = math.floor((left + right)/2 - (bottom-top)/2)
        right = left + (bottom - top)
    else:
        top = math.floor((top + bottom)/2 - (right-left)/2)
        bottom = top + (right - left)

    length = max(bottom - top + 1, right- left + 1)
    scale_factor = length / BOX_SIZE
    new_image = np.zeros((BOX_SIZE,BOX_SIZE))
    for i,row in enumerate(new_image):
        for j,_ in enumerate(row):
            old_i = int(max(0,min(top + np.round(scale_factor*i,decimals=0),27)))
            old_j = int(max(0,min(left + np.round(scale_factor*j),27)))
            new_image[i][j] = image[old_i][old_j]
    return new_image

def center_by_mass(image):
    m,n = image.shape
    image = fit_to_box(image)
    box_size = len(image)
    col_weighted = image*np.arange(1,box_size+1)
    row_weighted = np.transpose(np.transpose(image)*np.arange(1,box_size+1))
    
    #calculat mean position of pixels along each axis
    c_row= np.sum(np.sum(row_weighted,axis=0)) // np.sum(image) - 1
    c_col =  np.sum(np.sum(col_weighted,axis=0)) // np.sum(image) - 1

    row_shift = int(c_row - (box_size//2 -1))
    col_shift =  int(c_col - (box_size//2 - 1) )

    new_image = np.zeros((m,n))

    top,left = (m-box_size)//2-row_shift, (n-box_size)//2-col_shift
    bottom,right = top+box_size,left+box_size
    top_loss = -1*min(0,top)
    left_loss = -1*min(0,left)
    bottom_loss = -1*min(0,m-bottom)
    right_loss = -1*min(0,n-right)
    new_image[max(0,top):min(m,bottom),max(0,left):min(n,right)] = image[0+top_loss:box_size - bottom_loss,0+left_loss:box_size-right_loss]
    return new_image

def display(event,pixels):
    plt.imshow(np.transpose(center_by_mass(pixels)),'gray')
    plt.show()

def fill_pixels(event,canvas,pixels):
    x,y = event.x // 10 * 10, event.y // 10 * 10
    canvas.create_rectangle(x-10,y-10,x+10,y+10,fill="#000")
    i,j = event.x // 10, event.y // 10
    pixels[i-1:i+2,j-1:j+2] = 1



def print_prediction(pixels,model,label):
    pixels = center_by_mass(pixels)
    probabilities = model.predict(
        np.array([np.transpose(pixels)])
        )[0]
    probabilities[7] += 0.5
    probabilities[9] += 0.2
    prediction = np.argmax(probabilities)
    label["text"] = f"{prediction}"
    for i,num in enumerate(probabilities):
        print(i,num)

def reset(canvas,pixels):
    canvas.delete("all")
    pixels[:,:] = 0


