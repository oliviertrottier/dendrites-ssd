import time, sys
from .augmentations import SSDAugmentation


class Dict2struct(dict):
    """ Dictionary subclass whose entries can be accessed by attributes
        (as well as normally).
    """

    def __init__(self, *args, **kwargs):
        super(Dict2struct, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def convert(data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return Dict2struct({key: Dict2struct.convert(data[key])
                                for key in data})


def countdown(t):
    t_init = t
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)

        if t < t_init:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        print(timeformat, end="", flush=True)
        time.sleep(1)
        t -= 1
