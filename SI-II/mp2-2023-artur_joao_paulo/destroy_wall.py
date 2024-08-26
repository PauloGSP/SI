import math


class dest:


    # Given 2 points return the distance between them.
    @staticmethod
    def distance_points(point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    # Given one list of distances and one function, return the index of the closest position.
    @staticmethod
    def close_position(distances):
        small = distances[0]
        index = 0

        for ind, dist in enumerate(distances):
            if dist < small:
                small = dist
                index = ind

        return index

