class Coordinate:
    def __init__(self, x, y, score=-1):
        self.x = x
        self.y = y
        self.score = score

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_score(self):
        return self.score


class PersonData:
    def __init__(self, idx):
        self.idx = idx
        self.coordinates = []

    def append(self, coord):
        self.coordinates.append(coord)

    def get_coordinates(self):
        return self.coordinates
