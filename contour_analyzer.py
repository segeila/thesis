import cv2
from skimage.feature import blob_doh
from skimage.exposure import equalize_hist
from skimage.measure import find_contours
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_valid(sizes, labels, stats, centroids):
    """Finds coordinates of a centroid of a contour.

    Parameters
    ----------
    param1 : list  
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    tuple
        X and Y coordinates of a centroid

    """

    valid_mask = sizes > 30
    valid_indexes = [i for i, x in enumerate(valid_mask) if x]
    
    valid_labels = [x if x in valid_indexes else 0 for x in labels.flatten()]
    valid_labels = np.reshape(valid_labels, (labels.shape))

    valid_stats = [i for ind, i in enumerate(stats) if ind in valid_indexes]

    valid_centroids = [i for ind, i in enumerate(centroids) if ind in valid_indexes]

    return valid_labels, valid_stats, valid_centroids

def get_centroid(contour):
    """Finds coordinates of a centroid of a contour.

    Parameters
    ----------
    param1 : list  
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    tuple
        X and Y coordinates of a centroid

    """
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

def get_valid_contours(contours, max_length, min_length):
    """Filters out contours which are too small or too big.

    Parameters
    ----------
    param1 : list  
        Each element is a contour.
    param2 : int
        Max perimeter of a contour.
    param3 : int
        Min perimeter of a contour

    Returns
    -------
    list
        A subset of an input list which satisfies given limits.

    """
    valid_contours = []
    for c in contours:
        if (len(c) > min_length and len(c) <= max_length):
            valid_contours.append(c.astype(np.float32))
    return valid_contours

def countour_to_bbox(contour):
    """Finds a straight bounding box for a contour.

    Parameters
    ----------
    param1 : list  
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    tuple : int
        X and Y coordinates of a top-left point of a bounding box, its width and height.

    """
    x,y,w,h = cv2.boundingRect(contour)
    return x,y,w,h

def get_area(contour):
    """Finds an area of a contour.

    Parameters
    ----------
    param1 : list  
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    float 
        Area of a contour.

    """
    M = cv2.moments(contour)
    area = M['m00']
    return area

def get_circularity(perimeter, area):
    """Calculates circularity of a contour.

    Parameters
    ----------
    param1 : float  
        Perimeter of a contour.
    
    param1 : float  
        Area of a contour.

    Returns
    -------
    float
        Circularity of a contour.

    """
    circularity = np.power(perimeter, 2)/(4*np.pi*area)
    return circularity

def check_convexity(contour):
    """Checks if a contour is convex.

    Parameters
    ----------
    param1 : list  
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    bool
        True if a contour is convex, False otherwise.

    """
    return cv2.isContourConvex(contour)

def get_perimeter(contour):
    """Calculates closed and open perimeter of a contour.

    Parameters
    ----------
    param1 : list  
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    tuple : float
        First element is a closed perimeter, second element is an open perimeter.

    """
    perimeter_closed = cv2.arcLength(contour, True)
    perimeter_open = cv2.arcLength(contour, False)
    return perimeter_closed, perimeter_open

def get_convex_hull(contour):
    """Finds a convex hull of a contour (smallest convex set that contains contour).

    Parameters
    ----------
    param1 : list  
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    list 
         Each element is a list of X and Y coordinates of a point in a convex hull.

    """
    hull = cv2.convexHull(contour)
    return hull

def get_aspect_ratio(width, height):
    """Finds a convex hull of a contour (smallest convex set that contains contour).

    Parameters
    ----------
    param1 : list  
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    list
        Each element is a list of X and Y coordinates of a point in a convex hull.

    """
    aspect_ratio = float(width)/float(height)
    return aspect_ratio

def get_extent(area, width, height):
    """Calculates a ratio of a contour area to its bounding rectangle area.

    Parameters
    ----------
    param1 : float  
        Area of a contour.
    param2 : int  
        Width of a bounding box of a contour.
    param3 : int 
        Height of a bounding box of a contour.

    Returns
    -------
    float
        Ratio of a contour area to its bounding rectangle area.

    """
    extent = area/(width*height)
    return extent

def get_solidity(area, hull):
    """Calculates a ratio of a contour area to its convex hull area.

    Parameters
    ----------
    param1 : float  
        Area of a contour.
    param2 : list  
        Each element is a list of X and Y coordinates of a point in a convex hull.

    Returns
    -------
    float
        Ratio of a contour area to its convex hull area.

    """
    hull_area = get_area(hull)
    solidity = area/hull_area
    return solidity

def get_equivalent_diameter(area):
    """Calculates a diameter of a circle whose area is the same as a contour area.

    Parameters
    ----------
    param1 : float  
        Area of a contour.

    Returns
    -------
    float
        Diameter of a circle whose area is the same as a contour area.

    """
    equivalent_diameter = np.sqrt(4*area/np.pi)
    return equivalent_diameter

def get_orientation(contour):
    """Calculates an angle at which a contour is directed.

    Parameters
    ----------
    param1 : list 
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    int
        Angle at which a contour is directed.

    """
    (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
    return angle

def contour_to_mask(contour, shape):
    """Converts a contour to a binary mask.

    Parameters
    ----------
    param1 : list 
        Each element is a list of X and Y coordinates of a point in a contour.
    param2 : tuple
        First element is a height of a result mask, second element is a width of a result mask.

    Returns
    -------
    ndarray
        2D array which stores a binary mask.

    """
    contour = np.array(contour).astype(np.int32)
    contour = np.array([[y,x] for x,y in contour])
    mask = np.zeros(shape,np.uint8)
    cv2.drawContours(mask,[contour],0,255,-1)
    return mask

def get_mean_intensity(contour):
    """Calculates average intensity within a contour in a grayscale mode.

    Parameters
    ----------
    param1 : list 
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    float
        Average intensity within a contour.

    """
    mean_intensity = 0
    mask = contour_to_mask(contour, img.shape)
    mean_intensity, _, _, _ = cv2.mean(img, mask = mask.astype(np.uint8))
    return mean_intensity

def draw_rectangle(img, bbox):
    """Draws a white rectangle on top of an input image.

    Parameters
    ----------
    param1 : ndarray 
        2D array which stores an image on which a rectangle should be drawn.
    param2 : tuple
        X and Y coordinates of a top-left point of a rectangle, its width and height.

    Returns
    -------
    ndarray
        2D array which stores an input image with a rectangle drawn on it.

    """
    x,y,h,w = bbox
    cv2.rectangle(img,(y,x),(y+h,x+w),(255,255,255),-1)
    return img

def get_intensity_ratio(contour, bbox):
    """Calculates an average intensity of an upper quarter of a contour and an average intensity of the bottom three quarters
       of a contour. Usually a person's head is located in the upper part and hotter than the rest of a body so its average
       intensity should be higher. 

    Parameters
    ----------
    param1 : list
        Each element is a list of X and Y coordinates of a point in a contour.
    param2 : tuple
        X and Y coordinates of a top-left point of a bounding box, its width and height.

    Returns
    -------
    tuple : float
        First element is an average intensity of an upper part, second element is an average intensity of a bottom part of a contour.

    """
    intensity_ratio = 0

    y,x,h,w = bbox

    #Calculate a point at which contour will be split into two parts
    top_height = h//4
    bottom_height = h - top_height

    #GEt coordinates for an upper and bottom part
    top_bbox = (y,x,w,top_height)
    bottom_bbox = (y+top_height, x, w, h-top_height)

    #Create an empty black image
    full_mask = contour_to_mask(contour, img.shape)

    #Create an upper part of a contour
    top_mask = np.zeros(img.shape, np.uint8)
    top_mask = draw_rectangle(top_mask, top_bbox)
    top_region = np.bitwise_and(full_mask, top_mask)

    #Create a bottom part of a contour
    bottom_mask = np.zeros(img.shape, np.uint8)
    bottom_mask = draw_rectangle(bottom_mask, bottom_bbox)
    bottom_region = np.bitwise_and(full_mask, bottom_mask)

    #Calculate intensity for each part
    intensity_top, _, _, _ = cv2.mean(img, mask = top_region.astype(np.uint8))
    intensity_bottom, _, _, _ = cv2.mean(img, mask = bottom_region.astype(np.uint8))

    return intensity_top, intensity_bottom

def get_hull_ratio(len_contour, len_hull):
    """Calculates a ratio between amount of points in a contour against amount of points in its convex hull.

    Parameters
    ----------
    param1 : int 
        Amount of points in a contour.
    param1 : int 
        Amount of points in a contour's convex hull.

    Returns
    -------
    float
        Ratio of amount of points in a contour to amount of points in its convex hull.

    """
    ratio = (float(len_hull)/len_contour)*100
    return ratio

def check_protrusion(bbox, shape):
    """Checks if contour is completely inside an image or not.

    Parameters
    ----------
    param1 : tuple 
        X and Y coordinates of a top-left point of a bounding box, its width and height.
    param2 : shape
        Height and width of an image.

    Returns
    -------
    bool
        True if some parts of a contour are outside an image, False otherwise.

    """
    
    protrusion = False
    x,y,w,h = bbox
    if x==0 or y==0 or x+w==shape[1] or y+h==shape[0]:
        return True
    return protrusion

def check_closeness(perimeter):
    """Checks if contour is closed or not. Compares its closed and open perimeters.

    Parameters
    ----------
    param1 : tuple 
        Closed and open perimeters of a contour.

    Returns
    -------
    bool
        True if contour is closed, False otherwise.

    """
    if perimeter[0] == perimeter[1]:
        return True
    return False

def visualize_contour_metadata(contour_metadata, img):
    """Outputs an original image with a contour, its centroid and bounding box plotted on top of an image.

    Parameters
    ----------
    param1 : dict
        Dictionary which stores metadata about a contour.
    param2 : ndarray
        2D array which stores an input image.
    """
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

    ax.plot(contour_metadata['points'][:, 1], contour_metadata['points'][:, 0], linewidth=2)
    #ax.plot(contour_metadata['hull'][:, :, 1], contour_metadata['hull'][:, :, 0], linewidth=2)
    ax.plot(contour_metadata['centroid'][1], contour_metadata['centroid'][0], 'go')

    rect = patches.Rectangle((contour_metadata['bbox'][1],contour_metadata['bbox'][0]),contour_metadata['bbox'][3],contour_metadata['bbox'][2],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def print_metadata(contour_metadata):
    """Check if contour is completely inside an image or not.

    Parameters
    ----------
    param1 : dict
        Dictionary which stores metadata about a contour extracted during its analysis.

    """
    print('Contour consist of %d points' % len(contour_metadata['points']))
    print('Area %f/%f' % (contour_metadata['area'],(contour_metadata['area']/(120*160)*100))+'%')
    print('Perimeter %.3f' % contour_metadata['perimeter'])
    print('Area to perimeter ratio %.3f' % contour_metadata['ratio'])
    print('Width to height ratio %.3f' % contour_metadata['aspect_ratio'])
    print('Extent %.2f' % (contour_metadata['extent']*100) + '%') 
    print('Solidity %.2f' % contour_metadata['solidity'])
    print('Equivalent diameter %.3f' % contour_metadata['equivalent_diameter'])
    print('Convex? %s' % contour_metadata['convex'])
    print('Circularity %.3f' % contour_metadata['circularity'])
    print('Amount of points in contour\'s hull %d' % len(contour_metadata['hull']) )
    print('Orientation angle %.0f' % contour_metadata['orientation'])
    print('Mean intensity %.2f' % contour_metadata['mean_intensity'])
    print('Hull ratio %.2f' % contour_metadata['hull_ratio'] + '%')
    print('Protrusion %s' % contour_metadata['protrusion'])
    print('Closeness %s' % contour_metadata['closeness'])
    print('Intensity top region %.2f, bottom region %.2f' % (contour_metadata['intensity'][0], contour_metadata['intensity'][1]))

def analyze_contour(contour):
    """Extracts various features of a contour and stores them into a dictionary.

    Parameters
    ----------
    param1 : list 
        Each element is a list of X and Y coordinates of a point in a contour.

    Returns
    -------
    dict
        Dictionary that stores information about a contour.

    """
    contour = contour.astype(np.float32)
    contour_metadata = {}
    
    area = get_area(contour)
    perimeter = get_perimeter(contour)
    hull = get_convex_hull(contour)
    bbox = countour_to_bbox(contour)

    contour_metadata['points'] = contour
    contour_metadata['area'] = area
    contour_metadata['perimeter'] = perimeter[0]
    contour_metadata['ratio'] = area/perimeter[0]
    contour_metadata['bbox'] = bbox
    contour_metadata['aspect_ratio'] = get_aspect_ratio(bbox[2], bbox[3])
    contour_metadata['circularity'] = get_circularity(perimeter[0], area)
    contour_metadata['extent'] = get_extent(area, bbox[2], bbox[3]) 
    contour_metadata['hull'] = hull 
    contour_metadata['solidity'] = get_solidity(area, hull)
    contour_metadata['equivalent_diameter'] = get_equivalent_diameter(area)
    contour_metadata['convex'] = check_convexity(contour)
    contour_metadata['centroid'] = get_centroid(contour)
    contour_metadata['orientation'] = get_orientation(contour)
    contour_metadata['mean_intensity'] = get_mean_intensity(contour)
    contour_metadata['hull_ratio'] = get_hull_ratio(len(contour), len(hull))
    contour_metadata['protrusion'] = check_protrusion(bbox, img.shape)
    contour_metadata['closeness'] = check_closeness(perimeter)
    contour_metadata['intensity'] = get_intensity_ratio(contour, bbox)

    return contour_metadata

#Test feature extraction functions

#Read, blur, threshold image.
img = cv2.imread('blob.jpg', 0)
img = cv2.GaussianBlur(img, (3, 3), 0)
ret, otsu = cv2.threshold(img,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Find valid contours
contours = find_contours(img, 0.8)
valid_contours = get_valid_contours(contours, 150, 15)
print('Amount of valid contours', len(valid_contours))

#Analyze contour
contour_meta = analyze_contour(valid_contours[5])
print_metadata(contour_meta)

#Visualize contour at a given index, print out information about contour
visualize_contour_metadata(contour_meta, img)
