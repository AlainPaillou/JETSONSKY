#Interface to ZWO ASI motorized filter wheel USB.

import ctypes as c
import logging
import os
import six
import sys
import traceback


__author__ = 'Alain Paillou'
__version__ = '0.0.01'



def get_num_efw():
    return zwoefwlib.EFWGetNum()

def _get_ID(id_):
    num = c.c_int()
    r = zwoefwlib.EFWGetID(id_, num)
    if r:
        raise zwo_errors[r]
    return num.value

def _open_efw(id_):
    r = zwoefwlib.EFWOpen(id_)
    if r:
        raise zwo_errors[r]
    return

def _close_efw(id_):
    r = zwoefwlib.EFWClose(id_)
    if r:
        raise zwo_errors[r]
    return

def _get_position(id_):
    position = c.c_int()
    r = zwoefwlib.EFWGetPosition(id_, position)
    if r:
        raise zwo_errors[r]
    return position.value

def _set_position(id_, position):
    r = zwoefwlib.EFWSetPosition(id_, position)
    if r:
        raise zwo_errors[r]
    return

def _get_direction(id_):
    direction = c.c_bool()
    r = zwoefwlib.EFWGetDirection(id_, direction)
    if r:
        raise zwo_errors[r]
    return direction.value

def _set_direction(id_, direction):
    r = zwoefwlib.EFWSetDirection(id_, direction)
    if r:
        raise zwo_errors[r]
    return

def _get_product_ids():
    return zwoefwlib.EFWGetProductIDs()


def _get_efw_property(id_):
    prop = _EFW_INFO()
    r = zwoefwlib.EFWGetProperty(id_, prop)
    if r:
        raise zwo_errors[r]
    return prop.get_dict()



def list_EFW():
    """Retrieves model names of all connected ZWO filter wheels. Type :class:`list` of :class:`str`."""
    r = []
    for id_ in range(get_num_efw()):
        r.append(_get_efw_property(id_)['Name'])
    return r


class ZWO_Error(Exception):
    """Exception class for errors returned from the :mod:`zwoasi` module."""
    def __init__(self, message):
        Exception.__init__(self, message)


class ZWO_IOError(ZWO_Error):
    """Exception class for all errors returned from the ASI SDK library."""
    def __init__(self, message, error_code=None):
        ZWO_Error.__init__(self, message)
        self.error_code = error_code


class ZWO_CaptureError(ZWO_Error):
    """Exception class for when :func:`Camera.capture()` fails."""
    def __init__(self, message, exposure_status=None):
        ZWO_Error.__init__(self, message)
        self.exposure_status = exposure_status


class EFW(object):
    #Representation of ZWO ASI EFW.
    def __init__(self, id_):
        if isinstance(id_, int):
            if id_ >= get_num_efw() or id_ < 0:
                raise IndexError('Invalid id')
        elif isinstance(id_, six.string_types):
            # Find first matching EFW model
            found = False
            for n in range(get_num_efw()):
                model = _get_efw_property(n)['Name']
                if model in (id_, 'ZWO EFW ' + id_):
                    found = True
                    id_ = n
                    break
            if not found:
                raise ValueError('Could not find EFW model %s' % id_)

        else:
            raise TypeError('Unknown type for id')

        self.id = id_
        try:
            _open_efw(id_)
            self.closed = False
        except Exception:
            self.closed = True
            _close_efw(id_)
            logger.error('could not open EFW ' + str(id_))
            logger.debug(traceback.format_exc())
            raise
            
    def __del__(self):
            self.close()
            
    def get_efw_property(self):
        return _get_efw_property(self.id)

    def get_position(self):
        return _get_position(self.id)

    def set_position(self, position):
        _set_position(self.id, position)

    def get_direction(self):
        return _get_direction(self.id)

    def set_direction(self, direction):
        _set_direction(self.id, direction)
         
    def close(self):
        """Close the EFW in the ASI library.
        The destructor will automatically close the camera if it has not already been closed."""
        try:
            _close_efw(self.id)
        finally:
            self.closed = True

    def open(self):
        """Open the EFW in the ASI library.
        The destructor will automatically close the camera if it has not already been closed."""
        try:
            _open_efw(self.id)
        finally:
            self.closed = False

    def get_id(self):
        return _get_id(self.id)
    
    def get_product_ids(self):
        return _get_product_ids()


class _EFW_INFO(c.Structure):
    _fields_ = [
        ('ID', c.c_int),
        ('Name', c.c_char * 64),
        ('slotNum', c.c_int),
    ]

    def get_dict(self):
        r = {}
        for k, _ in self._fields_:
            v = getattr(self, k)
            if sys.version_info[0] >= 3 and isinstance(v, bytes):
                v = v.decode()
            r[k] = v
        del r['Unused']
        return r



def init(library_file=None):
    global zwoefwlib

    if zwoefwlib is not None:
        raise ZWO_Error('Library already initialized')

    if library_file is None:
        # PC Windows 10 version
        zwolib_filename = 'EFW_filter'
        libpath = "D:/Astro/Prgm/Python/ZWO/Lib"
        ext = '.dll'

        # Raspberry Pi Linux version
        #zwolib_filename = 'ASICamera2'
        #libpath = "/home/pi/Alain/Python/ZWO/lib/armv7/"
        #ext = '.so'


    zwoefwlib = c.cdll.LoadLibrary(library_file)

    zwoefwlib.EFWGetNum.argtypes = []
    zwoefwlib.EFWGetNum.restype = c.c_int

    zwoefwlib.EFWGetID.argtypes = [c.c_int]
    zwoefwlib.EFWGetID.restype = c.c_int

    zwoefwlib.EFWGetProperty.argtypes = [c.c_int,c.POINTER(_EFW_INFO)]
    zwoefwlib.EFWGetProperty.restype = c.c_int

    zwoefwlib.EFWOpen.argtypes = [c.c_int]
    zwoefwlib.EFWOpen.restype = c.c_int

    zwoefwlib.EFWClose.argtypes = [c.c_int]
    zwoefwlib.EFWClose.restype = c.c_int

    zwoefwlib.EFWGetPosition.argtypes = [c.c_int, c.POINTER(c.c_int)]
    zwoefwlib.EFWGetPosition.restype = c.c_int

    zwoefwlib.EFWSetPosition.argtypes = [c.c_int, c.c_int]
    zwoefwlib.EFWSetPosition.restype = c.c_int

    zwoefwlib.EFWSetDirection.argtypes = [c.c_int, c.c_bool]
    zwoefwlib.EFWSetDirection.restype = c.c_int

    zwoefwlib.EFWGetDirection.argtypes = [c.c_int, c.POINTER(c.c_bool)]
    zwoefwlib.EFWGetDirection.restype = c.c_int

    zwoefwlib.EFWGetProductIDs.argtypes = [c.POINTER(c.c_int)]
    zwoefwlib.EFWGetProductIDs.restype = c.c_int

logger = logging.getLogger(__name__)


# Mapping of error numbers to exceptions. Zero is used for success.
zwo_errors = [None,
              ZWO_IOError('Invalid index', 1),
              ZWO_IOError('Invalid ID', 2),
              ZWO_IOError('Invalid value', 3),
              ZWO_IOError('EFW closed', 4),
              ZWO_IOError('EFW removed', 5),
              ZWO_IOError('EFW moving', 6),
              ZWO_IOError('EFW general error', 7),
              ZWO_IOError('EFW error closed', 8),
              ZWO_IOError('EFW error end', -1),
              ]

# User must call init() before first use
zwoefwlib = None
