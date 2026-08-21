"""
    serfilereader
    jean-baptiste.butet ashashiwa@gmail.com 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    To be quick : this library can only be used by free/open softwares GPL compas.
"""

import numpy as np
import cv2, copy   
#from astropy.io import fits

 
#########tests purpose#########
import functools,time


def time_it(func):
    """Timestamp decorator for dedicated functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()                 
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        mlsec = repr(elapsed).split('.')[1][:3]
        readable = time.strftime("%H:%M:%S.{}".format(mlsec), time.gmtime(elapsed))
        print('Function "{}": {} sec'.format(func.__name__, readable))
        return result
    return wrapper

################################

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class Serfile():
    """
    Serfile provide a list of methods to use serfiles in python
    Compatible with V3 specifitations :  http://www.grischa-hahn.homepage.t-online.de/astro/ser/SER%20Doc%20V3b.pdf
    
    usage : serfile = Serfile(filename, NEW) NEW is a boolean if you want to create a NEW File
    
    Methods : 
    Public :
    .read() :           return a numpy frame,  position
    .getHeader() :      return SER file header in a dictionnary
    .readFrameAtPos(n:int) : return frame number n.
    .dateFrameAtPos(n:int) : return UTC DATE of File if possible else -1.
    .getLength() :      return number of frame in SER file.
    .getWidth() :       return width of a frame
    .getHeight() :      return height of a frame
    .getCurrentPosition():       return current cursor value
    .setCurrentPosition(n:int) : set the cursor at position n. Return 0 if OK, -1 if something's wrong. Began at 0
    .saveFit(filename) : save as FIT file filename frame at "cursor". Return (OK_Flag, filename)
    .savePng(filename) : save as PNG file filename frame at "cursor". Return (OK_Flag, filename)
    .nextFrame() :        add 1 to cursor i.e. go to next frame. Return current cursor value. -1 if the end of th video.
    .previousFrame() :  remove 1 to cursor i.e. go to previous frame. Return current cursor value. -1 if it's from first frame..
    .testFile(filneame) : test if filename exist. RAISE : FileNotFoundError
    .getName()          return name of SER file.
    .addFrame(frame) : add a frame at the end of file
    
    Private : 
    ._readExistingHeader() :    read header and return it in a dictionnary
    ._savePng_cv2( filename, datas)
    
     Header
    .createNewHeader(header)
    .setImageWidth(int)
    .setImageHeight(int)
    .setObserver(str)
    .setTelescope(str)
    .setInstrument(str)
    .setDateTime(int)
    .setDateTimeUTC(int)
    .setFileID(str)
    .setLuID(0)
    .setColorID(0)
    .setPixelDepthPerPlane(int)
    .setLittleEndian(int)
    
    
    Attributes: 
    Private : 
    self._cursor :      current position of video frame.
    self._offset  :     number of Bytes from beginning of file
    self._hdr_fits :    header of FIT file.
    self._currentFrame : numpy array containing frame at self._cursor
    self._width :       width of a frame
    self._height :      height of a frame
    self._frameDimension : product of width*height. Number of pixels of a frame.
    self._debug
    self._nameOfSerfile : name of file
    self._debug         : Debug Flag
    self._header        : SER file header in a dictionnary
    self._bytesPerPixels : number of bytes per pixels (depends on "colorId"
    
    Public : 
    
    
    """
    
    def __init__(self, name_of_serfile, NEW=False, header = None):
        self._nameOfSerfile = name_of_serfile
        
        self._debug = True
        self._trail = []
        if not NEW : 
            "" if self.testFile(self._nameOfSerfile) else self.quit()
            self._header, readOk, trail = self._readExistingHeader()
            if not readOk : 
                raise InputError("SERFile Error", "SER file doesn't look like a good SER File...")
            self._length = self._header.get('FrameCount', -1)
            self._frameDimension=self._header['ImageWidth']*self._header['ImageHeight'] #frame dimension, nb of Bytes
            self._width = self._header['ImageWidth']
            self._height = self._header['ImageHeight']
            
            if trail : 
                self._trail = self.readTrailFromHeader()
        elif header is None : 
            self.createNewHeader()
        self._cursor = 0
        self._currentFrame = np.array([])
        self._hdr_fits=""
        
    
    def dateFrameAtPos(self,n):
        return -1 if len(self._trail)==0 or n>(self.getLength()-1) else  self._trail[n]
    
    def readTrailFromHeader(self):
        """read header and return a liste of dates: [date1, date2,..,date_n] Each date is the frame's date."""
        
        trail = []
        offset = self._header['ImageHeight'] * self._header['ImageWidth'] * self._header['FrameCount'] * self._bytesPerPixels + 178
        with open(self._nameOfSerfile, 'rb') as file:
            for i in range(self.getLength()):
                file.seek(offset)
                trail.append(np.fromfile(file, dtype='uint64', count=1)[0])
                offset+=8    
        return trail
    
    def getName():
        return self._nameOfSerfile
    
    def quit(self):
        return -1
    
    def testFile(self,fileToOpen):
        with open(fileToOpen): pass
        
    def getHeader(self):
        return self._header
    
    def read(self):
        """read a frame and move forward
        In : None
        
        Out: frame, position
        
        """
        frame = self.readFrameAtPos(self._cursor)
        ret = self.nextFrame()
        return frame, ret
    
    def _readExistingHeader(self):
        """specifications : http://www.grischa-hahn.homepage.t-online.de/astro/ser/SER%20Doc%20V3b.pdf
        Be careful with case. First letter is an uppercase.
        Read header of of serfile.
        Return : Dict, Bool
        
        ColorId Documentation : 
        Content:MONO= 0
        BAYER_RGGB= 8
        BAYER_GRBG= 9
        BAYER_GBRG= 10
        BAYER_BGGR= 11
        BAYER_CYYM= 16
        BAYER_YCMY= 17
        BAYER_YMCY= 18
        BAYER_MYYC= 19
        RGB= 100
        BGR= 10
        """
        header = {}
        self._header={}
        readOK = True
        trail=False
        try : 
            with open(self._nameOfSerfile, 'rb') as file:
                #NB : don't use offset from np.fromFile to keep compatibility with all numpy versions
                FileID = np.fromfile(file, dtype='int8', count=14).tobytes().decode()
                header['FileID'] = FileID.strip()
                offset=14
                
                file.seek(offset)
                
                LuID = np.fromfile(file, dtype='uint32', count=1)[0]
                header['LuID'] = LuID
                offset += 4
                
                file.seek(offset)
            
                ColorID = np.fromfile(file, dtype='uint32', count=1)[0]
                header['ColorID'] = ColorID
                offset += 4
                
                file.seek(offset)
                
                LittleEndian = np.fromfile(file, dtype='uint32', count=1)[0]
                header['LittleEndian'] = LittleEndian
                offset += 4
                
                file.seek(offset)
                
                ImageWidth = np.fromfile(file, dtype='uint32', count=1)[0]
                header['ImageWidth'] = ImageWidth
                offset += 4
                
                file.seek(offset)
                
                ImageHeight = np.fromfile(file, dtype='uint32', count=1)[0]
                header['ImageHeight'] = ImageHeight
                offset += 4
                
                file.seek(offset)
                
                PixelDepthPerPlane = np.fromfile(file, dtype='uint32', count=1)[0]
                header['PixelDepthPerPlane'] = PixelDepthPerPlane
                offset += 4
                
                file.seek(offset)
                
                FrameCount = np.fromfile(file, dtype='uint32', count=1)[0]
                header['FrameCount'] = FrameCount
                offset += 4
                
                file.seek(offset)
                
                Observer = np.fromfile(file, dtype='int8', count=40).tobytes().decode()
                header['Observer'] = Observer.strip()
                offset+=40
                
                file.seek(offset)
                
                Instrument = np.fromfile(file, dtype='int8', count=40).tobytes().decode()
                header['Instrument'] = Instrument.strip()
                offset+=40
                
                file.seek(offset)
                
                Telescope = np.fromfile(file, dtype='int8', count=40).tobytes().decode()
                header['Telescope'] = Telescope.strip()
                offset+=40
                
                file.seek(offset)
                
                DateTime = np.fromfile(file, dtype='uint64', count=1)[0]
                header['DateTime'] = DateTime
                offset += 8
                
                file.seek(offset)
                
                DateTimeUTC = np.fromfile(file, dtype='uint64', count=1)[0]
                header['DateTimeUTC'] = DateTimeUTC
            
            
                file.seek(0,2)
        
                #####Handle ColorID#####
                if (ColorID <= 19):
                    NumberOfPlanes = 1
                else:
                    NumberOfPlanes = 3
                
                if (PixelDepthPerPlane <= 8):
                    BytesPerPixel = NumberOfPlanes
                else:
                    BytesPerPixel = 2 * NumberOfPlanes
                
                self._bytesPerPixels = BytesPerPixel
                #######################
                
                lengthWithoutTrail = ImageHeight * ImageWidth * FrameCount * BytesPerPixel + 178
                lengthWithTrail = ImageHeight * ImageWidth * FrameCount * BytesPerPixel + 178 + 8*FrameCount #SPECIFICATION : trail contain 1 date per frame.
                readOK = True
                trail = False
                if int(file.tell()) == int(lengthWithoutTrail) : 
                    trail = False
                elif int(file.tell()) == int(lengthWithTrail) :
                    trail = True
                
        except IndexError: 
            readOK=False
            
        return header, readOK, trail
    
    
    def getCurrentFrame(self):
        return self.readFrameAtPos(self._cursor)
    
    def readFrameAtPos(self,n):
        """return seek to offset 178 + n* self._frameDimension * self._bytesPerPixels bytes 
        and return self._frameDimension * self._bytesPerPixels bytes. 

        """
        if n<self._length : 
            with open(self._nameOfSerfile, 'rb') as file:
                frame = np.array([])
                offset = 178+n*self._frameDimension*self._bytesPerPixels
                file.seek(offset)
                if self._header['PixelDepthPerPlane']==8 :
                    frame=np.fromfile(file, dtype='uint8',count=self._frameDimension)
                else :
                    frame=np.fromfile(file, dtype='uint16',count=self._frameDimension)
#                frame=np.reshape(frame,(self._height,self._width))
                print("Height : ",self._height)
                print("Width : ",self._width)
                shape = [self._height,self._width]
                frame = frame.reshape(shape)
                self._currentFrame = frame
                return frame
        return -1
    
    def getLength(self):
        return self._header.get('FrameCount', -1)
    
    def getWidth(self):
        return self._header.get('ImageWidth', -1)
    
    def getHeight(self):
        return self._header.get('ImageHeight', -1)

    def getpixeldepth(self):
        return self._header.get('PixelDepthPerPlane', -1)
    
    def getCurrentPosition(self):
        return self._cursor
    
    def setCurrentPosition(self, n):
        #TODO test with type else return -1
        self._cursor = n
        try: #when reading, frames exists. So can be read. But when creating a frame, cursor is updated, without a frame that cause an error
            self.getCurrentFrame() 
        except ValueError: pass
        return self._cursor
    
    def createFitsHeader(self):
        hdr= fits.Header()
        hdr['SIMPLE']='T'
        hdr['BITPIX']=32
        hdr['NAXIS']=2
        hdr['NAXIS1']=self._header['ImageWidth']
        hdr['NAXIS2']=self._header['ImageHeight']
        hdr['BZERO']=0
        hdr['BSCALE']=1
        hdr['BIN1']=1
        hdr['BIN2']=1
        hdr['EXPTIME']=0
        return hdr
    
#    def saveFit(self, filename):
        #TODO use of astropyio https://docs.astropy.org/en/stable/io/fits/
#        if not 'fit' in filename.split('.')[-1].lower() : 
#            filename+='.fit'
#        if self._hdr_fits=="" : 
#            self._hdr_fits = self.createFitsHeader()
#        if len(self._currentFrame)==0 : 
#            self.read()
#        DiskHDU=fits.PrimaryHDU(self._currentFrame,header=self._hdr_fits)
#        DiskHDU.writeto(filename,overwrite=True)
#        return True, filename
    
    def savePng(self, filename):
        if not 'png' in filename.split('.')[-1].lower() : 
            filename+='.png'
        #On 16 bits, it possible to have somme issue. So theses lines transform 16bits in 8 bits.
        if len(self._currentFrame)==0:
            self.read()
        index_maximum = np.amax(self._currentFrame)
        datas = 256/index_maximum*self._currentFrame.astype(int)
        #return self._savePng_python(filename, datas)
        return self._savePng_cv2(filename, datas)
    
    def _savePng_python(self, filename, datas): #2 min for 3800 pictures
        #TODO : Doesn't work
        png.from_array(datas, 'L').save(filename)


    def _savePng_cv2(self, filename, datas): #8 seconds for 3800 pictures
        """save currentFrame in a PNG file"""
        #TODO : find a pythonic way to do
        return cv2.imwrite(filename, datas), filename
    
    def nextFrame(self):
        """add one to cursor. return actual frame position. -1 if nothing can be done"""
        if self._cursor<self._length : 
            self.setCurrentPosition(self._cursor+1)
        else : 
            return -1
        return self._cursor
    
    def previousFrame(self):
        """remove one to cursor. return actual frame position. -1 if nothing can be done"""
        if self._cursor>0 : 
            self.setCurrentPosition(self._cursor-1)
        else : 
            return -1
        return self._cursor
    
    def createNewHeader(self, header_=None):
        """Create SER header from an existing one. (or not)
        
        arguments : a dictionnary with all keys to fill SER file header
        
        return : None
        """
        #DOC: 'FileID', 'LuID','ColorID','LittleEndian', 'ImageWidth','ImageHeight','PixelDepthPerPlane', 'FrameCount', 'Observer','Instrument','Telescope','DateTime','DateTimeUTC'
        if header_==None : 
            header={}
        else: 
            header=copy.deepcopy(header_)
        keys_offset = {'FileID':(0,'uint8',14,True),
                       'LuID':(14, 'uint32', 1, False),
                       'ColorID':(18,'uint32',1,False ),
                       'LittleEndian':(22, 'uint32',1,False),  #have to be user defined
                       'ImageWidth':(26, 'uint32',1,False),    #have to be user defined
                       'ImageHeight':(30, 'uint32',1,False),   #have to be user defined
                       'PixelDepthPerPlane':(34,'uint32',1,False), #have to be user defined
                       'FrameCount':(38, 'uint32',1,False),    #will be marke at zero
                       'Observer':(42, 'int8', 40,True),
                       'Instrument': (82, 'int8', 40,True),
                       'Telescope': (122,'int8',40,True),
                       'DateTime': (162, 'uint64', 1, False),
                       'DateTimeUTC': (170,'uint64',1, False)}
        
        #write 178 Bytes with 0.
        with open(self._nameOfSerfile, 'wb',0) as _file : 
            _file.seek(0)
            for i in range(178):
                _file.write(np.int8(0))

        ##Test consistency
        #for key in keys_offset.keys() : 
            #test = header.get(key)
            #if test==None and not ('ImageHeight' in key or 'ImageWidth' in key or 'PixelDepthPerPlane' in key or 'FrameCount' in key):
                #raise ValueError("header error, miss some key")

        header['FrameCount'] = 0
        
        for header_key in keys_offset.keys():
            try : 
                self._updateHeader(header_key, header[header_key])
            except KeyError :
                #print('Key : ', header_key, "is not present")
                pass
        
        
        self._header, readOk, trail = self._readExistingHeader()        

    
    def _updateHeader(self, key, value):
        """update header and read it again to verify consistency"""
        keys_offset = {'FileID':(0,'uint8',14,True),
                       'LuID':(14, 'uint32', 1, False),
                       'ColorID':(18,'uint32',1,False ),
                       'LittleEndian':(22, 'uint32',1,False),
                       'ImageWidth':(26, 'uint32',1,False),
                       'ImageHeight':(30, 'uint32',1,False),
                       'PixelDepthPerPlane':(34,'uint32',1,False),
                       'FrameCount':(38, 'uint32',1,False),
                       'Observer':(42, 'int8', 40,True),
                       'Instrument': (82, 'int8', 40,True),
                       'Telescope': (122,'int8',40,True),
                       'DateTime': (162, 'uint64', 1, False),
                       'DateTimeUTC': (170,'uint64',1, False)}
        
        with open(self._nameOfSerfile, 'r+b',0) as _file : 
            offset=keys_offset[key][0]

            _file.seek(offset)
            if keys_offset[key][3]: #str
                while len(value)<keys_offset[key][2] :
                    value+=' '
                value = value.encode('ascii')
            else : #numeric
                if keys_offset[key][1]=='uint32':
                    value = np.uint32(value)
                elif keys_offset[key][1]=='uint64':
                    value = np.uint64(value)  
            _file.write(value)
        self._header, readOk, trail = self._readExistingHeader()   
        if readOk : 
            
            self._height = self._header.get('ImageHeight')
            self._width = self._header.get('ImageWidth')
            self._length = self._header.get('FrameCount')
            if self._header.get('ImageHeight') and self._header.get('ImageWidth'):
                self._frameDimension = self._height * self._width
    
#    def createFitsHeaderFromFitsFile(fitsFile):
        """read fits header and fill a new SER header with it
        
        In : a string containing fits file to read
        
        Out: header built.
        """
        
#        hdul = fits.open(fitsFile)
#        self.createNewHeader()
#        self._updateHeader('DateTime', hdul[0].header['DATE'])
#        self._updateHeader('FileID', 'LUCAM-RECORDER')
#        self._updateHeader('ColorID', 0)
#        self._updateHeader('LittleEndian', 0)
#        self._updateHeader('ImageHeight', hdul[0].header['NAXIS2'])
#        self._updateHeader('ImageWidth', hdul[0].header['NAXIS1'])
#        self._updateHeader('PixelDepthPerPlane', hdul[0].header['BITPIX']//16)
#        self._updateHeader('FrameCount', 0)
#        self._updateHeader('Observer', hdul[0].header['OBSERVER'])
#        self._updateHeader('Instrument', "")
#        self._updateHeader('Telescope', hdul[0].header['EXPTIME'])
#        self._updateHeader('DateTimeUTC', hdul[0].header['DATE'])
        
    
    def addFrame(self, frame):
        """add frame, add one to length. If it is the first frame, set height and width and frame_dimension
        In : numpy array containing datas
        """
        #try : 
            #self._frameDimension
        #except : 
            #self._frameDimension=
        try : 
            if self._header['FrameCount']==0 : #first frame
                self._updateHeader('ImageHeight',frame.shape[0])
                self._updateHeader('ImageWidth',frame.shape[1])
                self._height = self._header.get('ImageHeight')
                self._width = self._header.get('ImageWidth')
                self._frameDimension = self._height * self._width
            ### TODO : vérfiier taille de l'image 
            with open(self._nameOfSerfile, 'r+b',0) as _file : 
                _file.seek(178+self._cursor*self._frameDimension*self._bytesPerPixels)
                _file.write(frame)
                self.setCurrentPosition(self._cursor+1)
                self._length = self._length+1
                self._updateHeader('FrameCount', self._length)
        
        except IndexError:
                raise IndexError("tuple index out of range, may be Width and height are false. Shape :  %s"%(frame.shape))
            
    
    def setImageHeight(self, height):
        self._updateHeader('ImageHeight',height)
    
    def setImageWidth(self, width):
        self._updateHeader('ImageWidth',width)
        
    def setObserver(self, observer):
        self._updateHeader('Observer',observer)
        
    def setTelescope(self, telescope):
        self._updateHeader('Telescope',telescope)
        
    def setInstrument(self, instrument):
        self._updateHeader('Instrument',instrument)
        
    def setDateTime(self, datetime):
        self._updateHeader('DateTime',datetime)
        
    def setDateTimeUTC(self, datetimeutc):
        self._updateHeader('DateTimeUTC',datetimeutc)
        
    def setFileID(self, fileid):
        self._updateHeader('FileID',fileid)
        
    def setLuID(self,luid):
        self._updateHeader('LuID',luid)
        
    def setColorID(self,colorid):
        self._updateHeader('ColorID',colorid)
        
    def setPixelDepthPerPlane(self, pixeldepthperplace):
        self._updateHeader('PixelDepthPerPlane',pixeldepthperplace)
        self._header, readOk, trail = self._readExistingHeader()   
        
    def setLittleEndian(self, littleendian):
        self._updateHeader('ImageWidth',littleendian)
        self._header, readOk, trail = self._readExistingHeader()   
        
@time_it
def extract_png(serfile):
    for i in range(serfile.getLength()):
        serfile.savePng(sys.argv[2].split('.')[0]+str(i)+'.png')
        serfile.nextFrame()
    print("extracted %s frames"%(serfile.getLength()))
          
@time_it
def extract_fit(serfile):
    for i in range(serfile.getLength()):
        serfile.saveFit(sys.argv[2].split('.')[0]+str(i)+'.fit')
        serfile.nextFrame()
    print("extracted %s frames"%(serfile.getLength()))

if __name__ == "__main__":
    # execute only if run as a script
    import sys
    print("usage : setfilesreader.py -f file.ser")
    extract_png_flag = False
    extract_fit_flag = False
    save_png_flag = False
    save_fit_flag = False
    new_ser_flag = False
    if len(sys.argv)==3 : 
        print("file read : ", sys.argv[2])
        fichier_ser = Serfile(sys.argv[2])
        print('header', fichier_ser.getHeader())
        print('length', fichier_ser.getLength())
        print('width', fichier_ser.getWidth())
        print('height',fichier_ser.getHeight())
        print('frame n°10', fichier_ser.readFrameAtPos(10))
        print('date of 150th frame', fichier_ser.dateFrameAtPos(150))
        print('move to 100th frame', fichier_ser.setCurrentPosition(100))
        print('saving fit', fichier_ser.saveFit("try_fit100.fit"))
        print('saving png',fichier_ser.savePng("try_fit100.png"))


        if save_png_flag :
            n = 900
            fichier_ser.setCurrentPosition(n)
            fichier_ser.savePng(sys.argv[2].split('.')[0]+str(n)+'.png')
            
        if save_fit_flag :
            n = 800
            fichier_ser.setCurrentPosition(n)
            fichier_ser.saveFit(sys.argv[2].split('.')[0]+str(n)+'.fit')
        
        if extract_png_flag : ##Long...
            extract_png(fichier_ser)
                
        if extract_fit_flag : 
             extract_fit(fichier_ser)
        
    if new_ser_flag : 
        serfile_ = Serfile("try_ser.ser", NEW=True)
        #test purpose####
        header = {'FileID': 'LUCAM-RECORDER', 'LuID': 0, 'ColorID': 0, 'LittleEndian': 0, 'ImageWidth': 88, 'ImageHeight': 896, 'PixelDepthPerPlane': 16, 'FrameCount': 3884, 'Observer': 'SHELYAK', 'Instrument': 'ASI=ZWO ASI290MM Minitemp=32.0', 'Telescope': 'fps=80.00gain=30exp=0.38', 'DateTime': 637425855694020000, 'DateTimeUTC': 637425855693860000}
        #################
        serfile_.createNewHeader()
        serfile_.setImageWidth(100)
        
        serfile_.setObserver("SHELYAK")
        serfile_.setTelescope("fps=80.00gain=30exp=0.38")
        serfile_.setInstrument("ASI=ZWO ASI290MM Minitemp=32.0")
        serfile_.setDateTime(637425855693860000)
        serfile_.setDateTimeUTC(637425855693860000)
        serfile_.setFileID("LUCAM-RECORDER")
        serfile_.setLuID(0)
        serfile_.setColorID(0)
        serfile_.setPixelDepthPerPlane(16)
        serfile_.setLittleEndian(0)
        serfile_.setImageHeight(100)
        serfile_.setImageWidth(100)
        
        #adding frames : 30 black square 100x100, 30 whit square
        frame_b = np.ones(10000, dtype=np.uint16)
        frame_b = frame_b.reshape(100,100)
        frame_w = np.ones(10000, dtype=np.uint16)*(2**16-1)
        frame_w = frame_w.reshape(100,100)
        
        print('date of 1st frame', serfile_.dateFrameAtPos(1))
        for i in range(40):
            for i in range(100):
                serfile_.addFrame(frame_b)
            for i in range(100):
                serfile_.addFrame(frame_w)

        print(serfile_.getHeader())
    
