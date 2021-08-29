import cv2
import numpy as np
import time
import random
from   threading import Thread
from matplotlib import pyplot as plt
from numpy.random import default_rng

bee = np.array([[0, 0, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 1, 0]])
                #[0, 0, 0, 0]])
block = np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]]);
loaf = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 1, 0, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0]]);

class Live:
    """
    Class that calculate live cellular automata
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.height = 600
        self.width = 800
        self.frame = np.zeros((self.height,self.width,3), dtype=np.uint8)
        self.stopped = False
        self.block_size = 16   
        
        print("Hello Live!")

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def put_block(x,y,image,blk):
        s = len(blk);
        img[y-s:y+s,x-s:x+s,0:3] = blk.copy()
    
    def b2i(self,tuple):
        new_tuple = ()
        for item in tuple:
            blk2 = np.zeros((self.block_size, self.block_size, 3), dtype=np.uint8)
            x, y = (round(self.block_size/2),round(self.block_size/2))
            s=round(len(item[1])/2) 
            blk2[x-s:x+s,y-s:y+s,:][ item == 1 ] = (255,0,0) 
            blk2[x-s:x+s,y-s:y+s,:][ item == 0 ] = (0,0,0)
            new_tuple += (blk2,)
            #print('blk.shape ' + str(blk2.shape))
        return new_tuple


    def inital_image(self):
        img = np.zeros((self.width,self.width,3), dtype=np.uint8)
        blk = np.zeros((self.block_size, self.block_size, 3), dtype=np.uint8)
        blk[::] = (255,0,0) 
        print(f'blk.shape = {blk.shape}')
        num_cells = round(self.width / self.block_size);
        s = round(self.block_size / 2)
        
        rng = default_rng()
        p_colors = [.3, .7]
        p_blocks = [.2, .3, .5]
        init_color = (255,0,0)        
        #return img


        block_types = self.b2i((bee, loaf, block))
        print(block_types)
        #return img
        for x in range(0,num_cells,1):
            for y in range(0,num_cells,1):
                c = (y * self.block_size + s, x * self.block_size + s )                
                #colors = ((0,0,255), (255,0,0))
                #rng.choice(a=block_types,p=p_colors)
                blk = rng.choice(a=block_types,p=p_blocks)                
                img[c[0]-s:c[0]+s,c[1]-s:c[1]+s,0:3] = blk.copy()
        
        #cv2.imshow("img", img)
        cv2.imwrite("live.png", img)
        print(num_cells)
        return img

    def next_gen(self, frame):
        pixels = np.zeros((self.height,self.width,3), dtype=np.uint8)

        neibours = [(x,y) for x in [1,0,-1] for y in [1,0,-1]]
        print(neibours)
        print(pixels)
        return frame

    def get(self):
        while not self.stopped:
            t0 = time.time_ns() 
            self.frame = self.next_gen(self.frame)
            self.read_frame_flag = False
            lastFrameTime = (time.time_ns() - t0) / (10 ** 9)
            print('Live:: lastFrameTime  ' + str(lastFrameTime))
            cv2.imshow("live window", self.frame)

            time.sleep(0.4) 
            ch = cv2.waitKey(1)
            if ch & 0xFF == ord('q'):
                break

                #print("get(self)")

    def set_britness(self, params):
        self.britness = params[0]
        self.params = params

    def decrase_brightness(self, img, value=30):
        #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #h, s, v = cv2.split(hsv)

        if value < 0:
            lim = value
            img[img < lim] = 0
            img[img >= lim] -= value
        else:
            lim = value
            img[img > lim] = 255 - value
            img[img <= lim] += value


        #final_hsv = cv2.merge((h, s, v))
        #img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img               

    def stop(self):
        self.stopped = True

live = Live(0)
#live.next_gen(live.frame)
img = live.inital_image()
cv2.imshow("live window", img)
ch = cv2.waitKey(-1)

live.stop()
cv2.destroyAllWindows()
#cv2.imshow("noise", noise)

