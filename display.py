import sys
import pygame
import time
from pygame.locals import *
import numpy as np
from threading import Thread

from settings import colors, neuronalType, neuronId, fftDisplayXLimits, recording_chunk_size, sampling_rate, displaySize


pygame.init()

COMMAND_PLAY = 0


class BallDisplay(pygame.Surface):
    def __init__(self, screenSize=0, imageSize=0, limits=[-80e-3,-40e-3],
                 bkgColor=(0, 0, 0, 0), EL=-60e-3, threshold=-50e-3,
                 **kwargs):
        super(BallDisplay, self).__init__((screenSize, screenSize), **kwargs)
        self.center = (int(round(screenSize/2)), int(round(screenSize/2)))
        self.imageSize = imageSize
        self.limits = limits
        self.bkgColor = bkgColor
        self.fill(self.bkgColor)
        
        self.baseValue = self.__v2Rad(EL)
        self.threshold = self.__v2Rad(threshold)
        self.show()
        
    def __v2Rad(self, v):
        si = int(max([self.imageSize/2. * np.abs((self.limits[0] - v))/np.diff(self.limits), 0]))
        return si
    
    def show(self, v=None):
        self.fill(self.bkgColor)
        pygame.draw.circle(self,  pygame.Color('green'), self.center, self.baseValue, 2)
        pygame.draw.circle(self, pygame.Color('red'), self.center, self.threshold, 2)
        if v:
            pygame.draw.circle(self, colors[neuronalType], self.center, self.__v2Rad(v))
        

class MembraneDisplay(pygame.Surface):
    def __init__(self, size=(0, 0), xpoints=500., ylim=[-80e-3, -45e-3],
                 baseValue=-60e-3,
                 bkgColor=(0, 0, 0, 0), lineColor=(0, 0, 255, 0), threshold=-50e-3,
                 **kwargs):
        super(MembraneDisplay, self).__init__(size, **kwargs)
        self.xpoints = xpoints
        self.ylim = np.array(ylim)
        self.bkgColor = bkgColor
        self.lineColor = lineColor
        self.baseValue = baseValue
        self.thresholdCoords = self.__coord2Px(np.array([0, xpoints-1]),
                                               np.array([threshold, threshold]))
        self.fill(self.bkgColor)
        self.xdata = np.arange(xpoints)
        self.ydata = np.ones(int(xpoints))*self.baseValue
        self.show()
        
    def __coord2Px(self, x, y):
        xCoords = x*self.get_width()/self.xpoints
        yCoords = self.get_height()-self.get_height()*(y-self.ylim[0])/np.diff(self.ylim)
        return list(zip(xCoords, yCoords))
    
    def addValue(self, v):
        if v == []:
            self.ydata = np.concatenate((self.ydata[1::], [self.baseValue]))
        else:
            self.ydata = np.concatenate((self.ydata[1::], [v]))
        self.show()
    
    def show(self):
        pxCoords = self.__coord2Px(self.xdata, self.ydata)
        self.fill(self.bkgColor)
        pygame.draw.aalines(self, pygame.Color('red'), False, self.thresholdCoords, 0)
        pygame.draw.lines(self, colors[neuronalType], False, pxCoords, 1)


class GraphDisplay(pygame.Surface):
    def __init__(self, size=(0, 0), xlim=[-1, 1], ylim=[-1, 1],
                 bkgColor=(0, 0, 0, 0),
                 **kwargs):
        super(GraphDisplay, self).__init__(size, **kwargs)
        self.xlim = np.array(xlim)
        self.ylim = np.array(ylim)
        self.bkgColor = bkgColor
        self.fill(self.bkgColor)
        
    def _coord2Px(self, x, y):
        xCoords = self.get_width()*(x-self.xlim[0])/np.diff(self.xlim)
        yCoords = self.get_height()-self.get_height()*(y-self.ylim[0])/np.diff(self.ylim)
        return list(zip(xCoords, yCoords))
    
    def plot(self, x, y):
        self.fill(self.bkgColor)
        pxCoords = self._coord2Px(x, y)
        pygame.draw.aalines(self, colors['membrane'], False, pxCoords)


class FFTDisplay(GraphDisplay):
    def __init__(self, types=None, intervals=None, **kwargs):
        super(FFTDisplay, self).__init__(**kwargs)
        self.__types = types
        self.__intervals = intervals
        self.__detected = np.zeros(len(intervals))
        self.__createFrequencySurf()
        
    def plot(self, x, y, detected_frequencies):
        self.fill(self.bkgColor)
        for k, val in enumerate(detected_frequencies):
            rect = self.__freqRects[k][0]
            col = pygame.Color('red') if val else self.__freqRects[k][1] 
            pygame.draw.rect(self, col, rect)
             
        pxCoords = self._coord2Px(x, y)
        pygame.draw.lines(self, colors['fft'], False, pxCoords, 1)
        
    def __createFrequencySurf(self):
        self.__freqRects = []
        self.__freqCols = []
        for k, intv in enumerate(self.__intervals):
            edgeCoords = self._coord2Px(np.log(intv), (self.ylim[0], self.ylim[1]))
            w = np.log(np.abs(np.diff(edgeCoords[0])))[0]
            print('-----', edgeCoords[0][0], 0, w, self.get_height())
            rect = pygame.Rect(edgeCoords[0][0], 0, w, self.get_height())
            self.__freqRects.append(
                (
                    rect,
                    colors[self.__types[k]]
                )
            )


class FullDisplay:
    SPIKE_COL = [255, 0, 0]
    SIZERATIO = [.3, .5, .2]

    def __init__(self, playedFrequency=None, frequencies=None, types=None, intervals=None,
                 startAudioCb=None, threshold=0, resting_potential=0, width=400, height=800):
        self.width = width
        self.height = height
        self.__hasSpiked = False
        self.__startAudio = None
        self.__fullscreen = False

        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Neuron '+str(neuronId)+' @ '+str(playedFrequency)+'Hz')
        
        self.fftRect = pygame.Rect(0, 0, self.width, self.height*self.SIZERATIO[0])
        self.fftDisplay = FFTDisplay(types, intervals,
                                     size=(self.width, self.height*self.SIZERATIO[0]),
                                     xlim=np.log(fftDisplayXLimits), ylim=[0, 300])
        self.display.blit(self.fftDisplay, self.fftRect)

        self.ballRect = pygame.Rect(0, self.height*self.SIZERATIO[0],
                                    self.width, self.height*self.SIZERATIO[1],
                                    limits=[-80e-3, threshold])
        self.ballDisplay = BallDisplay(screenSize=int(self.height*self.SIZERATIO[1]),
                                       imageSize=int(self.height*(self.SIZERATIO[1]-0.05)),
                                       limits=[-80e-3, threshold],
                                       threshold=threshold,
                                       EL=resting_potential)
        self.display.blit(self.ballDisplay, self.ballRect)
        
        self.membraneRect = pygame.Rect(0, self.height*(1-self.SIZERATIO[2]),
                                        self.width, self.height*self.SIZERATIO[2])
        self.membraneDisplay = MembraneDisplay(size=(self.width, self.height*self.SIZERATIO[2]),
                                               ylim=[-80e-3, threshold*.95])
        self.display.blit(self.membraneDisplay, self.membraneRect)

        frqs = np.arange(recording_chunk_size) / (recording_chunk_size/float(sampling_rate))
        self.__x_data = frqs[list(range(int(recording_chunk_size / 2)))]

    def setStartAudioCb(self, pyfunc):
        self.__startAudio = pyfunc
        
#    def setFftData(self,x,y):
#        self.fftDisplay.plot(np.log(x[x>0]),y[x>0])
#        
#    def setMembraneData(self,v):
#        self.membraneDisplay.addValue(v)
#        self.ballDisplay.show(v)
#        
#    def setSpiked(self,spiked):
#        self.__hasSpiked = spiked

    def update(self, has_fired, y=None, v=None, detected_freqs=None):
        self.fftDisplay.plot(np.log(self.__x_data[self.__x_data > 0]), y[self.__x_data > 0], detected_freqs)
        self.membraneDisplay.addValue(v)
        self.ballDisplay.show(v)
        self.draw(has_fired)
        
    def draw(self, has_fired):
        if has_fired:
            self.display.fill(colors[neuronalType])
        else:
            self.display.blit(self.fftDisplay, self.fftRect)
            self.display.blit(self.ballDisplay, self.ballRect)
            self.display.blit(self.membraneDisplay, self.membraneRect)
        pygame.display.flip()

    def get_keyboard_input(self):
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                sys.exit(0)
            elif event.type == pygame.locals.MOUSEBUTTONDOWN:
                return FullDisplay.COMMAND_PLAY
            elif event.type == pygame.locals.KEYDOWN:
                if event.dict['key'] == pygame.locals.K_p:
                    return FullDisplay.COMMAND_PLAY
                elif event.dict['key'] == pygame.locals.K_ESCAPE:
                    sys.exit(0)
                elif event.dict['key'] == pygame.locals.K_f:
                    self.__fullscreen = not self.__fullscreen
                    if self.__fullscreen:
                        pygame.display.set_mode(displaySize, pygame.locals.FULLSCREEN)
                    else:
                        pygame.display.set_mode(displaySize)
        
    def toggleFullscreen(self):
        pygame.display.toggle_fullscreen()
        
        
class FFTHandler(object):
    def __init__(self, display, rect):
        super(FFTHandler, self).__init__()
        self.__display = display
        self.__rect = rect
        self.surface = GraphDisplay((400, 700)) #ylim=[-70,-40]
#        self.screen = app.Screen(screen,(800,0),(400, 400))
#        self.surface.plot(np.linspace(-1,1,2000),2*np.random.rand(2000)-1)
        self.__server = None
        self.setupOSC()
        self.update()
        
    def setupOSC(self):
        if self.__server is not None:
            self.__server.close()
            self.__server = None
        server = OSC.OSCServer( ('127.0.0.1', simulationPars()['membranePort']) )
        ms = threading.Thread( target = server.serve_forever )
        ms.start()
        server.addMsgHandler("/fromNetwork",self.__fromNetwork)
        self.__server = server
        
    def update(self, v=[]):
        self.surface.addValue(v)
        self.__display.blit(self.surface,self.__rect)
        ''' THE LEAK DOES NOT OCCUR WHEN EVENTS INDUCE THE DISPLAY_UPDATE!!! '''
        pygame.display.update() 
#        pygame.display.flip()
        
    def __fromNetwork(self, addr, tags, data, source):
        print(data)
        self.update()


class Screen:
    def __init__(self,surface,surfPos,size):
        self.display = pygame.display.set_mode(size)
        self.surface = surface
        self.rect = pygame.Rect((surfPos[0],surfPos[1], 
                                 self.surface.get_width(), self.surface.get_height()))
        
        self.update()
    
    def update(self):
        self.display.blit(self.surface, self.rect.topleft)
        pygame.display.update()


class Display(Thread):
    def __init__(self,surfPos,screenSize,**kwargs):
        super(Display,self).__init__()
        self.graphDisplay = GraphDisplay(**kwargs)
        self.screen = Screen(self.graphDisplay,surfPos,screenSize)
        self.busy=False
        self.__isRunning = False
        self.xData = None
        self.yData = None
        self.__refreshS = .5
        
    def setData(self,x,y):
        self.xData = x
        self.yData = y
        
    def run(self):
        self.__isRunning = True
        now = time.time()
        self.plot()
        while self.__isRunning:
            
            while time.time()-now<self.__refreshS:
                pass
            self.plot()
            now = time.time()

    def stop(self):
        self.__isRunning = False
        
    def plot(self):
        print('plotting')
        self.graphDisplay.plotFitted(self.xData,self.yData)
        self.screen.update()


if __name__=='__main__':
    surface = GraphDisplay((300,300))
    screen = Screen(surface,(30,30),(400, 400))
#    display = pygame.display.set_mode()
    
    time.sleep(1)
    surface.plot(np.linspace(-1,1,2000),2*np.random.rand(2000)-1)
    screen.update()
#    display.blit(surf, my_rect.topleft)
#    pygame.display.update()
    xx = np.linspace(-1,1,20)
    for k in range(100):
        time.sleep(.01)
        yy = 2*np.random.rand(20)-1
        now = time.time()
        surface.plot(xx,yy)
        screen.update()
    #    display.blit(surf, my_rect.topleft)
    #    pygame.display.update()
#        print 'passed',time.time() - now
#    time.sleep(1)
#    while not pygame.event.wait().type in (QUIT, KEYDOWN):
#        pass
#    uwe = Display()