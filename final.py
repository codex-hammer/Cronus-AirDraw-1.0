import random
from socket import *
from shutil import copyfile
import numpy as np
import cv2
import os
#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint
import random

gameMode = False
toDraw=""
def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('localhost', 9000))
im=""
imName=""

def FM(img1_name, img2_name):
    img1 = cv2.imread(img1_name)
    c1 = img1[int(img1.shape[0]*0.145833):int(img1.shape[0]*0.85416),int(img1.shape[1]*0.04375):int(img1.shape[1]*0.953125)]
    img2 = cv2.imread(img2_name)
    c2 = img2[int(img2.shape[0]*0.145833):int(img2.shape[0]*0.85416),int(img2.shape[1]*0.04375):int(img2.shape[1]*0.953125)]
    c1l = c1[:, 0:int(c1.shape[1] * 0.5)]
    c1r = c1[:, int(c1.shape[1] * 0.5):c1.shape[1]]
    c2l = c2[:, 0:int(c2.shape[1] * 0.5)]
    c2r = c2[:, int(c2.shape[1] * 0.5):c2.shape[1]]
    
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(c1l, None)
    kp2, des2 = orb.detectAndCompute(c2l, None)
    kp3, des3 = orb.detectAndCompute(c1r, None)
    kp4, des4 = orb.detectAndCompute(c2r, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches1 = bf.match(des1, des2)
    matches2 = bf.match(des3, des4)
    # Sort them in the order of their distance.
    matches1 = sorted(matches1, key = lambda x:x.distance)
    matches2 = sorted(matches2, key = lambda x:x.distance)
    x=len(matches1)
    y=len(matches2)
    ans = (x + y) / 2
    # Draw first 10 matches.
    #img3 = cv2.drawMatches(c1l, kp1, c2l, kp2, matches1[:1000], None, flags=2)
    #img4 = cv2.drawMatches(c1r, kp3, c2r, kp4, matches2[:1000], None, flags=2)
    #plt.imshow(img3), plt.show()
    #plt.imshow(img4), plt.show()
    #print(ans)
    return ans

def randomImage():
    files = os.listdir("images")
    index = random.randrange(0, len(files))
    return files[index][:-13]
  
def compare(toDraw):
    drawn_img="xyz.png"
    ans=drawn_img
    x=os.listdir("images")

    score = {}
    cnt = {}
    #Initialise dictionary
    for i in x:
        cnt[i[:-13]] = 0
        score[i[:-13]] = 0
    
    #Iterate over images     
    for i in x:
        curr = i[:-13]
        k = FM("images/"+i,drawn_img)
        cnt[curr] = cnt[curr] + 1 
        score[curr] = score[curr] + k
        
    #Iterate over keys
    maxAVG = 0
    finalRes = {}
    final=""
    
    for i in cnt:
        res = score[i]/cnt[i]
        finalRes[i] = res
        if( maxAVG < res ):
            maxAVG = res
            final = i    

    if ( gameMode ):
        print (toDraw)
        res = finalRes[toDraw]/maxAVG
        res = res*100
        res = round(res,2)
        print (maxAVG)
        print (finalRes[toDraw])
        print (res)
        y = "score: "+ str(res)
        toDraw = randomImage()
        
        y = y +" it looks like "+final+", try:" + " " + toDraw
        print (y)
        return y,toDraw

    t = finalRes.items()
    for k,v in t:
        print (k,v)

        
    sorted_by_second = sorted(t, key=lambda t: t[1])

    totalAVG = 0
    for k,v in sorted_by_second[-2:]:
        totalAVG = totalAVG + v
        
    perc = (maxAVG/totalAVG)*100
    perc = round(perc,2)
    q = "We are "+str(perc)+"% sure that you've drawn a "+final
    mxx = 0
    im = ""
    for i in x:
        if (i[:-13] == final):
            im = i
            break
        
    i = cv2.imread('images/'+im)
    c1 = i[int(i.shape[0]*0.145833):int(i.shape[0]*0.85416),int(i.shape[1]*0.04375):int(i.shape[1]*0.953125)]
    imgplot = plt.imshow(c1)
    plt.show()
    return q,final

while True:
     rand = random.randint(0, 10)
     message, address = serverSocket.recvfrom(80000)
     message=message.decode()
     print (message)
     #message = "image"
     if message == "close":
          print(message)
          serverSocket.sendto("close".encode(), address)
          serverSocket.close()
          break
     elif message == "hello":
          print(message)
          serverSocket.sendto("hello".encode(), address)
     elif message == "True":
          gameMode = True
          toDraw = randomImage()
          y="Draw " + toDraw
          print (y)
          serverSocket.sendto(y.encode(),address)
     elif message == "False":
          gameMode = False
          toDraw = ""
     elif message == "image":
          if (gameMode):
              im,toDraw = compare(toDraw)
          else:
              im,imName = compare(toDraw)
          print (im)
          print (imName)
          serverSocket.sendto(im.encode(), address)
          #serverSocket.sendto("100".encode(), address)
     elif message == "1":
          x=random_with_N_digits(8)
          result="images/"+imName.lower()+"_"+str(x)+".png"
          print(result)
          copyfile("xyz.png",result)
     else:
          x = random_with_N_digits(8)
          result = "images/" + message +"_"+ str(x) + ".png"
          copyfile("xyz.png", result)
