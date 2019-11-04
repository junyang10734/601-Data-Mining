"""
RANSAC Algorithm Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to fit a line to the given points using RANSAC algorithm, and output
the names of inlier points and outlier points for the line.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
You can use the library random
Hint: It is recommended to record the two initial points each time, such that you will Not 
start from this two points in next iteration.
"""
import random


def solution(input_points, t, d, k):
    """
    :param input_points:
           t: t is the perpendicular distance threshold from a point to a line
           d: d is the number of nearby points required to assert a model fits well, you may not need this parameter
           k: k is the number of iteration times
           Note that, n for line should be 2
           (more information can be found on the page 90 of slides "Image Features and Matching")
    :return: inlier_points_name, outlier_points_name
    inlier_points_name and outlier_points_name is two list, each element of them is str type.
    For example: If 'a','b' is inlier_points and 'c' is outlier_point.
    the output should be two lists of ['a', 'b'], ['c'].
    Note that, these two lists should be non-empty.
    """
    # TODO: implement this function.

    inlier_points_names = []
    outlier_points_names = []
    num = len(input_points)
    i = 0
    pList = []
    final_err = 9999

    while i < k and i < num*(num-1)/2:
        # store inlier points and outlier points of the current iteration
        this_inlier_points = []
        this_outlier_points = []

        p = random.sample(input_points, 2)

        str1 = p[0]['name']+p[1]['name']
        str2 = p[1]['name']+p[0]['name']
        # randomly select two points, if the two points have been calculated before, randomly select the points again.
        if str1 not in pList and str2 not in pList:
            this_inlier_points.append(p[0]['name'])
            this_inlier_points.append(p[1]['name'])
        else:
            while str1 in pList or str2 in pList:
                p = random.sample(input_points, 2)
                str1 = p[0]['name']+p[1]['name']
                str2 = p[1]['name']+p[0]['name']
            this_inlier_points.append(p[0]['name'])
            this_inlier_points.append(p[1]['name'])

        pList.append((p[0]['name']+p[1]['name']))

        x1 = p[0]['value'][0]
        x2 = p[1]['value'][0]
        y1 = p[0]['value'][1]
        y2 = p[1]['value'][1]

        line_type = 0
        if x2 == x1:
            line_type = 1  # the line is parallel to y-axis
            m = x2
        elif y2 == y1:
            line_type = 2  # the line is parallel to x-axis
            m = y1

        this_dis = 0
        for item in input_points:
            if item not in p:
                x0 = item['value'][0]
                y0 = item['value'][1]
                if line_type == 0:
                    distance = getDis(x0, y0, x1, y1, x2, y2)
                elif line_type == 1:
                    distance = x0 - m
                else:
                    distance = y0 - m
                
                # judge whether it is an inlier point or a outlier point by distance 
                if abs(distance) > t:
                    this_outlier_points.append(item['name'])
                else:
                    this_inlier_points.append(item['name'])
                    this_dis += abs(distance)

        if (len(this_inlier_points) - 2) >= d:
            # calculate error, if current error less than final_err,
            # then set final_err, inlier_points_names and outlier_points_names as current data
            this_err = this_dis / (len(this_inlier_points) - 2)
            if this_err < final_err:
                inlier_points_names = this_inlier_points
                outlier_points_names = this_outlier_points
                final_err = this_err

        i = i+1

    inlier_points_names.sort()
    outlier_points_names.sort()
    return inlier_points_names, outlier_points_names
    raise NotImplementedError


# compute the distance from a point to a line
def getDis(pointx, pointy, linex1, liney1, linex2, liney2):
    a = liney2-liney1
    b = linex1-linex2
    c = linex2*liney1-linex1*liney2
    dis = (abs(a*pointx+b*pointy+c))/(pow(a*a+b*b, 0.5))
    return dis


if __name__ == "__main__":
    input_points = [{'name': 'a', 'value': (0.0, 1.0)}, {'name': 'b', 'value': (2.0, 1.0)},
                    {'name': 'c', 'value': (3.0, 1.0)}, {'name': 'd', 'value': (0.0, 3.0)},
                    {'name': 'e', 'value': (1.0, 2.0)}, {'name': 'f', 'value': (1.5, 1.5)},
                    {'name': 'g', 'value': (1.0, 1.0)}, {'name': 'h', 'value': (1.5, 2.0)}]

    t = 0.5
    d = 3
    k = 100

    inlier_points_name, outlier_points_name = solution(input_points, t, d, k)

    assert len(inlier_points_name) + len(outlier_points_name) == 8
    f = open('./results/task1_result.txt', 'w')
    f.write('inlier points: ')
    for inliers in inlier_points_name:
        f.write(inliers + ',')
    f.write('\n')
    f.write('outlier points: ')
    for outliers in outlier_points_name:
        f.write(outliers + ',')
    f.close()
