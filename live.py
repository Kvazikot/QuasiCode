import cv2
import numpy as np
import time, random, math
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
        self.height = 100
        self.width = 100
        self.rules = self.maze_rulez
        self.frame = np.zeros((self.height,self.width,3), dtype=np.uint8)
        self.stopped = False
        self.block_size = 16   
        self.frame = self.inital_image()
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
        #print(f'blk.shape = {blk.shape}')
        num_cells = round(self.width / self.block_size);
        s = round(self.block_size / 2)
        
        rng = default_rng()
        p_colors = [.3, .7]
        p_blocks = [.2, .3, .5]
        init_color = (255,0,0)        
        #return img
        print("inital_")


        block_types = self.b2i((bee, loaf, block))
        #return img
        for x in range(0,num_cells,1):
            for y in range(0,num_cells,1):
                c = (y * self.block_size + s, x * self.block_size + s )                
                #colors = ((0,0,255), (255,0,0))
                #rng.choice(a=block_types,p=p_colors)
                blk = rng.choice(a=block_types,p=p_blocks)                
                img[c[0]-s:c[0]+s,c[1]-s:c[1]+s,0:3] = blk.copy()
        
        
        
        print(num_cells)
        return img

    def maze_rulez(self, blk):
        if (blk[1,1] !=0):
            live = True
        else:
            live = False
        blk[1,1] = 0
        n_live_neib = np.count_nonzero(blk)                                
        return_flag = False
        if live and (n_live_neib == 3) or (n_live_neib == 2): #reproduction.
            #print(f"n_live_neib={n_live_neib}")
            return_flag = True
        if (not live) and (n_live_neib == 3): #survival
            return_flag = True
        return return_flag


    def next_gen(self, frame, rules):
        next_frame = np.zeros((self.width,self.width,3), dtype=np.uint8)
        next_frame = frame
        cell_color = (255,0,0)
        for x in range(1,self.width-2,1):
            for y in range(1,self.width-2,1):
                blk = frame[y-1:y+2,x-1:x+2,0:1]
                live = rules(blk)
                if live:
                    next_frame[y,x,:] = cell_color
                else:
                    next_frame[y,x,:] = (0,0,0)       
        return next_frame

    def get(self):
        while not self.stopped:
            t0 = time.time_ns() 
            resized = cv2.resize(self.frame, (self.frame.shape[1]*5,self.frame.shape[0]*5))
            cv2.imshow("live window", resized)
            self.frame = self.next_gen(self.frame, self.rules)
            self.read_frame_flag = False
            lastFrameTime = (time.time_ns() - t0) / (10 ** 9)
            print('Live:: lastFrameTime  ' + str(lastFrameTime))
            
            time.sleep(0.4) 
            ch = cv2.waitKey(-1)
            if ch & 0xFF == ord('q'):
                break

                #print("get(self)")

    def stop(self):
        self.stopped = True

live = Live(0)
live.rules = live.maze_rulez
img = live.inital_image()
live.get()
ch = cv2.waitKey(-1)

live.stop()
cv2.destroyAllWindows()
#cv2.imshow("noise", noise)

