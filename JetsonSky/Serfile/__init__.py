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
import cv2
 

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
                offset = 178+n*np.multiply(self._frameDimension, self._bytesPerPixels, dtype=object)
                file.seek(offset)
                if self._header['PixelDepthPerPlane']==8 :
                    frame=np.fromfile(file, dtype='uint8',count=self._frameDimension)
                else :
                    frame=np.fromfile(file, dtype='uint16',count=self._frameDimension)
#                frame=np.reshape(frame,(self._height,self._width))
#                print("Height : ",self._height)
#                print("Width : ",self._width)
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
        except ValueError : pass
        return self._cursor
    
    
