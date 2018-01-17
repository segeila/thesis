import cv2
from skimage.feature import blob_doh
from skimage.exposure import equalize_hist
from skimage.measure import find_contours
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Read, blur, threshold image. Show resulting image
img = cv2.imread('blob.jpg', 0)
img = cv2.GaussianBlur(img, (3, 3), 0)
ret, otsu = cv2.threshold(img,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

def extract_blobs(img):

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 5
    params.maxThreshold = 2000

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 15

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.1

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(255 - img)
    print(len(keypoints))

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array(
        []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", img_with_keypoints)
    cv2.waitKey(0)


def find_edges(img):
    edges = cv2.Canny(img, 50, 200)
    cv2.imshow('laplacian', edges)
    cv2.waitKey(0)

def find_contours_cv(img):
	im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	cv2.drawContours(img, contours, -1, (0,255,0), 3)
	cv2.imshow('edges', im2)
	cv2.waitKey(0)

def auto_canny(image, sigma = 0.33):
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	cv2.imshow('edges', edged)
	cv2.waitKey(0)
	return edged

def find_contours_2():
    contours = find_contours(otsu, 0.8)
    #print(len(contours))

    # Display the image and plot all contours found
    ''' fig, ax = plt.subplots()
    ax.imshow(otsu, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show() '''

    return contours

def connected_components(img):
    #CC_STAT_LEFT 	
    #The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.

    #CC_STAT_TOP 	
    #The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.

    #CC_STAT_WIDTH 	
    #The horizontal size of the bounding box.

    #CC_STAT_HEIGHT 	
    #The vertical size of the bounding box.

    #CC_STAT_AREA 	
    #The total area (in pixels) of the connected component.

    output = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)

    num_labels = output[0]

    #Fisrt label is for background, therefore all elements are zeros. Matrix the size of the input image where
    #each element has a value equal to its label.
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    print('Components found %d' % num_labels)
    return num_labels, labels, stats, centroids

def get_largest(labels, sizes):
    max_label = 1
    max_size = sizes[1]
    for i in range(2, num_labels):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(labels.shape)
    img2[labels == max_label] = 255
    cv2.imshow("Biggest component", img2)
    cv2.waitKey()

def draw_roi(img, labels, stats, centroids):
    for ind, stat in enumerate(stats[1:]):
        x = stat[0]
        y = stat[1]
        width = stat[2]
        height = stat[3]
        cv2.rectangle(img, (x,y), (x+width,y+height), (255,255,255), thickness=1, lineType=8, shift=0)
        cv2.circle(img, (int(centroids[ind][0]),int(centroids[ind][1])), 2, (255,255,255))
    cv2.imshow('rectangles', img)
    cv2.waitKey(0)

def get_valid(sizes, labels, stats, centroids):
    valid_mask = sizes > 30
    valid_indexes = [i for i, x in enumerate(valid_mask) if x]
    
    valid_labels = [x if x in valid_indexes else 0 for x in labels.flatten()]
    valid_labels = np.reshape(valid_labels, (labels.shape))

    valid_stats = [i for ind, i in enumerate(stats) if ind in valid_indexes]

    valid_centroids = [i for ind, i in enumerate(centroids) if ind in valid_indexes]

    return valid_labels, valid_stats, valid_centroids

def test_connected_components(img):
    num_labels, labels, stats, centroids = connected_components(otsu)
    sizes = stats[:, -1]
    labels, stats, centroids = get_valid(sizes, labels, stats, centroids)
    draw_roi (img, labels, stats, centroids)


def get_centroid(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

def draw_contours(contours):
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(otsu, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        #ax.plot(get_centroid(contour)[0], get_centroid(contour)[1], 'go')

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

contours = find_contours_2()


def get_valid_contours(contours, max_length, min_length):
    valid_contours = []
    for c in contours:
        if (len(c) > min_length and len(c) <= max_length):
            valid_contours.append(c)
    return valid_contours

valid_contours = get_valid_contours(contours, 150, 15)
for c in valid_contours:
    c.astype(np.float32)

print('Amount of valid contours', len(valid_contours))

def countour_to_bbox(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return x,y,w,h

def get_area(contour):
    #area = cv2.contourArea(contour)
    M = cv2.moments(contour)
    area = M['m00']
    return area

def get_circularity(perimeter, area):
    circularity = np.power(perimeter, 2)/(4*np.pi*area)
    return circularity

def check_convexity(contour):
    return cv2.isContourConvex(contour)

def get_perimeter(contour):
    perimeter_closed = cv2.arcLength(contour, True)
    perimeter_open = cv2.arcLength(contour, False)
    return perimeter_closed, perimeter_open

def get_convex_hull(contour):
    hull = cv2.convexHull(contour)
    return hull

def get_aspect_ratio(width, height):
    aspect_ratio = float(width)/float(height)
    return aspect_ratio

def get_extent(area, width, height):
    extent = area/(width*height)
    return extent

def get_solidity(area, hull):
    hull_area = get_area(hull)
    solidity = area/hull_area
    return solidity

def get_equivalent_diameter(area):
    equivalent_diameter = np.sqrt(4*area/np.pi)
    return equivalent_diameter

def get_orientation(contour):
    (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
    return angle

def contour_to_mask(contour):
    contour = np.array(contour).astype(np.int32)
    contour = np.array([[y,x] for x,y in contour])
    mask = np.zeros(img.shape,np.uint8)
    cv2.drawContours(mask,[contour],0,255,-1)
    return mask

def get_mean_intensity(contour):
    mean_intensity = 0
    mask = contour_to_mask(contour)
    cv2.imshow('Mean intensity mask', mask)
    cv2.waitKey(0)
    mean_intensity, _, _, _ = cv2.mean(img, mask = mask.astype(np.uint8))
    return mean_intensity

def draw_rectangle(img, bbox):
    x,y,h,w = bbox
    cv2.rectangle(img,(y,x),(y+h,x+w),(255,255,255),-1)
    return img

def get_intensity_ratio(contour, bbox):
    intensity_ratio = 0
    y,x,h,w = bbox

    top_height = h//4
    bottom_height = h - top_height

    top_bbox = (y,x,w,top_height)
    bottom_bbox = (y+top_height, x, w, h-top_height)

    full_mask = contour_to_mask(contour)

    top_mask = np.zeros(img.shape, np.uint8)
    top_mask = draw_rectangle(top_mask, top_bbox)
    top_region = np.bitwise_and(full_mask, top_mask)

    bottom_mask = np.zeros(img.shape, np.uint8)
    bottom_mask = draw_rectangle(bottom_mask, bottom_bbox)
    bottom_region = np.bitwise_and(full_mask, bottom_mask)

    intensity_top, _, _, _ = cv2.mean(img, mask = top_region.astype(np.uint8))
    intensity_bottom, _, _, _ = cv2.mean(img, mask = bottom_region.astype(np.uint8))
    print(intensity_top, intensity_bottom)
    intensity_ratio = intensity_top/intensity_bottom

    return intensity_top, intensity_bottom

def get_hull_ratio(len_contour, len_hull):
    ratio = (float(len_hull)/len_contour)*100
    return ratio

def check_protrusion(bbox):
    protrusion = False
    x,y,w,h = bbox
    if x==0 or y==0 or x+w==0 or y+h==0:
        return True
    return protrusion

def check_closeness(perimeter):
    if perimeter[0] == perimeter[1]:
        return True
    return False

def visualize_contour_metadata(contour_metadata, img):
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

    ax.plot(contour_metadata['points'][:, 1], contour_metadata['points'][:, 0], linewidth=2)
    #ax.plot(contour_metadata['hull'][:, :, 1], contour_metadata['hull'][:, :, 0], linewidth=2)
    ax.plot(contour_metadata['centroid'][1], contour_metadata['centroid'][0], 'go')

    rect = patches.Rectangle((contour_metadata['bbox'][1],contour_metadata['bbox'][0]),contour_metadata['bbox'][3],contour_metadata['bbox'][2],linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def print_metadata(contour_metadata):
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
    contour = contour.astype(np.float32)
    contour_metadata = {}
    
    area = get_area(contour)
    perimeter = get_perimeter(contour)
    hull = get_convex_hull(contour)
    bbox = countour_to_bbox(contour)
    print(bbox)
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
    contour_metadata['protrusion'] = check_protrusion(bbox)
    contour_metadata['closeness'] = check_closeness(perimeter)
    contour_metadata['intensity'] = get_intensity_ratio(contour, bbox)

    print_metadata(contour_metadata)

    return contour_metadata

visualize_contour_metadata(analyze_contour(valid_contours[5]), img)
