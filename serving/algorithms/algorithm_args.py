class Category(object):
    def __init__(self, id, name):
        self.id = id
        self.name = name
    

class AlgorithmArgs(object):
    def __init__(self):
        pass


class DetectionArgs(AlgorithmArgs):
    def __init__(self, confidence=0.0):
        self.confidence = confidence
        self.categories = None


class ClassificationArgs(AlgorithmArgs):
    def __init__(self, confidence=0.5):
        self.confidence = confidence
        self.categories = None