import cv2
import numpy as np
import copy
import os
import ast
import collections

# _____________________________ Help Functions ______________________________


"""
this function gets 2 points, calculates the line conneting them, and returns the set of the points that lay on the line
"""


def get_line(x1, y1, x2, y2):
    points = []
    is_steep = abs(y2 - y1) > abs(x2 - x1)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    delta_x = x2 - x1
    delta_y = abs(y2 - y1)
    error = int(delta_x / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if is_steep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= delta_y
        if error < 0:
            y += ystep
            error += delta_x
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


"""
this function operates Harris Corner Detection algorithm on img, and writes the corners found on corners_on_black.
(it also writes them on corners_on_img but this has only de-bug purposes) 
"""


def harris_corner_detection(img, corners_on_img, corners_on_black, free_parameter=0.04):
    # find Harris corners
    dst = cv2.cornerHarris(img, 2, 3, free_parameter)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img, np.float32(centroids), (5, 5), (-1, -1), criteria)
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    corners_on_img[res[:, 1], res[:, 0]] = [255, 255, 0]  # draw the corners on the image
    corners_on_black[res[:, 1], res[:, 0]] = [0, 0, 255]  # draw the corners on black image
    return corners, corners_on_img, corners_on_black


"""
this function operates Shi-Tomasi Corner Detection algorithm on img, and writes the corners found on corners_on_black.
(it also writes them on corners_on_img but this has only de-bug purposes) 
"""


def shi_tomasi_corner_detection(img, corners_on_img, corners_on_black):
    corners = cv2.goodFeaturesToTrack(img, 200, 0.001, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        corners_on_black[y, x] = [0, 0, 255]
        corners_on_img[y, x] = [255, 255, 255]
    return corners, corners_on_img, corners_on_black


"""
this function gets an edges-image, and a slope of a line between 0 to 1 and information whether deltha
 x or deltha y of the line is greater (the information comes in 'vertical' parameter.
 according to the slope and the vertical parameters, it operates erosion morphological transformation on the image
 in order to clean noise
"""


def calc_edges_from_slope(edges, vertical, slope):
    if slope <= 0.01:
        param = 10
    if slope <= 0.015:
        param = 9
    elif slope <= 0.02:
        param = 8
    elif slope <= 0.03:
        param = 7
    elif slope <= 0.06:
        param = 6
    elif slope <= 0.075:
        param = 5
    elif slope <= 0.09:
        param = 4
    else:
        param = 3
    param = 3
    if vertical == True:
        kernel = np.ones((param, 1), np.uint8)
    else:
        kernel = np.ones((1, param), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = remove_small_connected_components(edges, 2 * param)
    return edges


"""
this function gets a current tested combination, asseses its value, and returns the best combination found so far 
"""


def calc_best_points(edges, final_result, final_result_values, output):
    count_01 = count_12 = count_23 = count_30 = size_01 = size_12 = size_23 = size_30 = 0
    #####################################################################################
    points01 = get_line(output[0][1], output[0][0], output[1][1], output[1][0])
    for point in points01:
        size_01 = size_01 + 1
        if edge_around(edges[0], point[0], point[1], height, width):
            count_01 = count_01 + 1
    #####################################################################################
    points12 = get_line(output[2][1], output[2][0], output[1][1], output[1][0])
    for point in points12:
        size_12 = size_12 + 1
        if edge_around(edges[1], point[0], point[1], height, width):
            count_12 = count_12 + 1
    #####################################################################################
    points23 = get_line(output[2][1], output[2][0], output[3][1], output[3][0])
    for point in points23:
        size_23 = size_23 + 1
        if edge_around(edges[2], point[0], point[1], height, width):
            count_23 = count_23 + 1
    #####################################################################################
    points30 = get_line(output[3][1], output[3][0], output[0][1], output[0][0])
    for point in points30:
        size_30 = size_30 + 1
        if edge_around(edges[3], point[0], point[1], height, width):
            count_30 = count_30 + 1

    result01 = count_01 / size_01
    result12 = count_12 / size_12
    result23 = count_23 / size_23
    result30 = count_30 / size_30
    all_result = [result01, result12, result23, result30]

    if min(all_result) > min(final_result_values):
        for i in range(len(output)):
            final_result[i] = output[i]
            final_result_values[i] = all_result[i]


def is_legal(i, j, height, width):
    if (i < 0) or (i >= width):
        return False
    elif (j < 0) or (j >= height):
        return False
    else:
        return True


"""
this function gets a pixel and an edges-image and returns True iff one of the pixel's neighbors in the edges-image is white colored. 
"""


def edge_around(edges, i, j, height, width):
    for k in range(i - 1, i + 2):
        for l in range(j - 1, j + 2):
            if is_legal(k, l, width, height):
                if edges[k, l] == [255]:
                    return True
    return False


"""
this function removes small connected components from img. (small = size less tham param)
this is also an effort to reduce noise in the edges-image.
"""


def remove_small_connected_components(img, param):
    # find all your connected components (white blobs in your image)
    nb_components, out, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # ConnectedComponentsWithStats yields every separated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = param
    # your answer image
    img_temp = np.zeros(out.shape)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img_temp[out == i + 1] = 255
    return img_temp


# _____________________________ Main ______________________________

# Program works best with a data set folder which contains images and crop text
# Change path according to your PC path.

# Input Image
path = r"C:\Users\User\Desktop\Data_Set\images_Labeled\img_02214.jpg"
#path = "Data Set/img_00045.jpg"  # Amit's PC

img = cv2.imread(path, 1)
source_img = copy.copy(img)
height, width = img.shape[0:2]

# Scale Down Image if necessary
max_size = 1000000  # Minimum size for scale down
c = 0.5  # Scale down factor
scaled = (height * width) >= max_size
if scaled:
    img = cv2.resize(img, (int(width * c), int(height * c)), interpolation=cv2.INTER_CUBIC)

# Extract Data Points From Text File
name, ext = os.path.splitext(path)
data_sets = []
with open(name + '.txt') as fp:
    line = fp.readline()
    while line:
        data_sets.append(ast.literal_eval(line))
        line = fp.readline()
crop_points = []
for data_set in data_sets:
    crop = []
    for dict_obj in data_set:
        crop.append([dict_obj.get("x"), dict_obj.get("y")])
    crop_points.append(crop)

original_crop_points = copy.deepcopy(crop_points)
if scaled:
    for i in range(len(crop_points)):
        for j in range(len(crop_points[i])):
            crop_points[i][j][0] = round(crop_points[i][j][0] * c)
            crop_points[i][j][1] = round(crop_points[i][j][1] * c)


# Crop Image Using Input (X,Y) Dots given to us
contours = []
for crop_point in original_crop_points:
    contours.append(np.array(crop_point).astype(int))
img_with_crop = copy.copy(source_img)
for contour in contours:
    cv2.drawContours(img_with_crop, [contour], 0, (0, 0, 255), 1)

# Print Image [FOR DEBUG PURPOSES]
cv2.imshow('Image with Input Crop', img_with_crop)
cv2.imwrite('Input.png', img_with_crop)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# ____________________ Initialization For PHASE #1 ____________________

# Deep Copy of img and Convert to Gray
corners_on_img = copy.copy(img)
corners_on_black = np.zeros((height, width, 3), np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sharpened Gray Image
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
gray_sharpen = cv2.filter2D(gray, -1, kernel)

# ------------------------------------- RUN CODE ON BASIC GRAY IMAGE ---------------------------------------------------

# __________ Run On Canny #1 __________
edges = cv2.Canny(gray, 60, 180)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black, 0.08)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black, 0.25)
corners, corners_on_img, corners_on_black = shi_tomasi_corner_detection(edges, corners_on_img, corners_on_black)

# __________ Run On Canny #2 __________
edges = cv2.Canny(gray, 7, 25)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black, 0.08)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black, 0.25)
corners, corners_on_img, corners_on_black = shi_tomasi_corner_detection(edges, corners_on_img, corners_on_black)

# __________ Run On Gray Image __________
corners, corners_on_img, corners_on_black = harris_corner_detection(gray, corners_on_img, corners_on_black)
corners, corners_on_img, corners_on_black = harris_corner_detection(gray, corners_on_img, corners_on_black, 0.08)
corners, corners_on_img, corners_on_black = harris_corner_detection(gray, corners_on_img, corners_on_black, 0.25)
corners, corners_on_img, corners_on_black = shi_tomasi_corner_detection(gray, corners_on_img, corners_on_black)

# __________ Run On blurred Gray Image __________
blur = cv2.medianBlur(gray, 3)
corners, corners_on_img, corners_on_black = harris_corner_detection(blur, corners_on_img, corners_on_black)
corners, corners_on_img, corners_on_black = harris_corner_detection(blur, corners_on_img, corners_on_black, 0.08)
corners, corners_on_img, corners_on_black = harris_corner_detection(blur, corners_on_img, corners_on_black, 0.25)
corners, corners_on_img, corners_on_black = shi_tomasi_corner_detection(blur, corners_on_img, corners_on_black)

# ---------------------------------- RUN CODE ON SHARPENED GRAY IMAGE --------------------------------------------------

# __________ Run On Canny #1 __________
edges = cv2.Canny(gray_sharpen, 60, 180)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black, 0.08)
corners, corners_on_img, corners_on_black = shi_tomasi_corner_detection(edges, corners_on_img, corners_on_black)

# __________ Run On Canny #2 __________
edges = cv2.Canny(gray_sharpen, 10, 30)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black, 0.08)
corners, corners_on_img, corners_on_black = shi_tomasi_corner_detection(edges, corners_on_img, corners_on_black)

# __________ Run On Gray Sharpen Image __________
gray_sharpen = np.float32(gray_sharpen)
corners, corners_on_img, corners_on_black = harris_corner_detection(gray_sharpen, corners_on_img, corners_on_black)
corners, corners_on_img, corners_on_black = harris_corner_detection(gray_sharpen, corners_on_img, corners_on_black,
                                                                    0.08)
corners, corners_on_img, corners_on_black = shi_tomasi_corner_detection(gray_sharpen, corners_on_img, corners_on_black)

# ------------------------------------ RUN CODE ON BLURRED GRAY IMAGE --------------------------------------------------

# __________ Run On Canny #1 __________
blur = cv2.medianBlur(gray, 3)
edges = cv2.Canny(blur, 60, 180)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black, 0.08)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black, 0.25)
corners, corners_on_img, corners_on_black = shi_tomasi_corner_detection(edges, corners_on_img, corners_on_black)

# __________ Run On Canny #2 __________
blur = cv2.medianBlur(gray, 3)
edges = cv2.Canny(blur, 5, 25)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black, 0.08)
corners, corners_on_img, corners_on_black = harris_corner_detection(edges, corners_on_img, corners_on_black, 0.25)
corners, corners_on_img, corners_on_black = shi_tomasi_corner_detection(edges, corners_on_img, corners_on_black)

# ---------------------------------------- PRINT FOUND CORNERS ---------------------------------------------------------

# Print Image Corners [FOR DEBUG PURPOSES]
"""
cv2.imshow("corners", corners_on_img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
cv2.imwrite("corners.png", corners_on_img)
"""

# ____________________ PHASE #2 ____________________

candidates = []
final_result = []
final_result_values = []
for crop_point in crop_points:
    candidates.append([dict(), dict(), dict(), dict()])
    final_result.append(np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
    final_result_values.append([-1, -1, -1, -1])

for point_index in range(len(candidates)):
    for k in range(len(candidates[point_index])):
        [i, j] = crop_points[point_index][k]
        for x in range(-30, 30):
            for y in range(-30, 30):
                if x + i < 0 or y + j < 0 or x + i >= width or y + j >= height:
                    continue
                if np.all(corners_on_black[y + j, x + i] == [0, 0, 255]):
                    candidates[point_index][k].setdefault((x + i, y + j), pow(x, 2) + pow(y, 2))

        sorted_x = sorted(candidates[point_index][k].items(), key=lambda kv: kv[1])
        candidates[point_index][k] = collections.OrderedDict(sorted_x)

# Prepare Edges For Scoring System
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img, (3, 3), 0)
blur = cv2.bilateralFilter(blur, 7, 75, 75)
edges = cv2.Canny(blur, 4, 14)


# Print Image Score Edges [FOR DEBUG PURPOSES]
"""
cv2.imshow("edges for score before changes!", edges)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
"""

for point_index in range(len(candidates)):
    verticals = [False, False, False, False]
    slopes = [0, 0, 0, 0]
    deltha_x01 = abs(crop_points[point_index][0][0] - crop_points[point_index][1][0])
    deltha_y01 = abs(crop_points[point_index][0][1] - crop_points[point_index][1][1])
    deltha_x12 = abs(crop_points[point_index][1][0] - crop_points[point_index][2][0])
    deltha_y12 = abs(crop_points[point_index][1][1] - crop_points[point_index][2][1])
    deltha_x23 = abs(crop_points[point_index][2][0] - crop_points[point_index][3][0])
    deltha_y23 = abs(crop_points[point_index][2][1] - crop_points[point_index][3][1])
    deltha_x30 = abs(crop_points[point_index][3][0] - crop_points[point_index][0][0])
    deltha_y30 = abs(crop_points[point_index][3][1] - crop_points[point_index][0][1])

    if deltha_x01 > deltha_y01:
        verticals[0] = False
        slopes[0] = deltha_y01 / deltha_x01
    else:
        verticals[0] = True
        slopes[0] = deltha_x01 / deltha_y01

    if deltha_x12 > deltha_y12:
        verticals[1] = False
        slopes[1] = deltha_y12 / deltha_x12
    else:
        verticals[1] = True
        slopes[1] = deltha_x12 / deltha_y12

    if deltha_x23 > deltha_y23:
        verticals[2] = False
        slopes[2] = deltha_y23 / deltha_x23
    else:
        verticals[2] = True
        slopes[2] = deltha_x23 / deltha_y23

    if deltha_x30 > deltha_y30:
        verticals[3] = False
        slopes[3] = deltha_y30 / deltha_x30
    else:
        verticals[3] = True
        slopes[3] = deltha_x30 / deltha_y30
    edges_arr = []
    edges_arr.append(calc_edges_from_slope(edges, verticals[0], slopes[0]))
    # cv2.imshow("edges01", calc_edges_from_slope(edges, verticals[0], slopes[0]))
    edges_arr.append(calc_edges_from_slope(edges, verticals[1], slopes[1]))
    # cv2.imshow("edges12", calc_edges_from_slope(edges, verticals[1], slopes[1]))
    edges_arr.append(calc_edges_from_slope(edges, verticals[2], slopes[2]))
    # cv2.imshow("edges23", calc_edges_from_slope(edges, verticals[2], slopes[2]))
    edges_arr.append(calc_edges_from_slope(edges, verticals[3], slopes[3]))
    # cv2.imshow("edges30", calc_edges_from_slope(edges, verticals[3], slopes[3]))
    edges_arr = np.array(edges_arr)

    blank_img = np.zeros((height, width, 3), np.uint8)
    output = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    counter = 1  # Round Index
    number_of_variations = 9
    number_of_rounds = min(len(candidates[point_index][0]), number_of_variations) \
                       * min(len(candidates[point_index][1]), number_of_variations) \
                       * min(len(candidates[point_index][2]), number_of_variations) \
                       * min(len(candidates[point_index][3]), number_of_variations)  # Number of variations checked
    for i in range(min(len(candidates[point_index][0]), number_of_variations)):
        for j in range(min(len(candidates[point_index][1]), number_of_variations)):
            for z in range(min(len(candidates[point_index][2]), number_of_variations)):
                for w in range(min(len(candidates[point_index][3]), number_of_variations)):
                    output[0] = list(candidates[point_index][0].keys())[i]
                    output[1] = list(candidates[point_index][1].keys())[j]
                    output[2] = list(candidates[point_index][2].keys())[z]
                    output[3] = list(candidates[point_index][3].keys())[w]

                    calc_best_points(edges_arr, final_result[point_index], final_result_values[point_index], output)
                    if counter % 10 == 0:
                        print(str(point_index + 1) + "# CROP:  " + str(counter) + " possibility out of "
                              + str(number_of_rounds))
                    counter = counter + 1

# Crop Image Using new (X,Y) Dots in result_points
contours = []
contours_scaled_img = []
for result in final_result:
    if scaled:
        contours.append(np.array(result // c).astype(int))
        contours_scaled_img.append(np.array(result).astype(int))
    else:
        contours.append(np.array(result).astype(int))

# Draw new contours in the Image
for contour in contours:
    cv2.drawContours(source_img, [contour], 0, (0, 0, 255), 1)
if scaled:
    for contour in contours_scaled_img:
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
    # Print Output Crop Image [FOR DEBUG PURPOSES]
    cv2.imshow('Final Output [Scaled Down]', img)

# Save Output Image With adjusted crop
cv2.imwrite('Output.png', source_img)

# Print Output Crop Image [FOR DEBUG PURPOSES]
cv2.imshow('Final Output', source_img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
