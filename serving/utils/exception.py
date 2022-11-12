class RequestArgsError(Exception):
    code = 410
    message = "input error"


class AIServingError(Exception):
    code = 420
    message = "api serving error"


class TFServingError(Exception):
    code = 430
    message = "model serving error"