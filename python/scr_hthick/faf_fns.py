import math
import statistics
import numpy as np
import cv2 as cv

def flatten(img, flat, a, c, ar, n):
    dim = img.shape[0:2]
    samp2 = [0]*n
    Barr=[[0]]*n

    flat0arr = [[0 for i in range(dim[1])] for j in range(dim[0])]
    flat1arr = [[0 for i in range(dim[1])] for j in range(dim[0])]

    ra = max(c[0], abs(dim[0]-c[0]))
    rb = max(c[1], abs(dim[1]-c[1]))
    samp2[0] = ra*ra + ar*ar*rb*rb
    samp2[0]*= 0.99
    samp2[0]/= n

    for i in range(1,n):
        samp2[i] = samp2[i-1]+samp2[0]
    
    for i in range(n):
        r = [math.sqrt(samp2[i]), math.sqrt(samp2[i])]
        ellpts = ellipsePoints(c,r)

        ellvals = []
        for point in ellpts:
            if point[0]>=0 and point[1]>=0 and point[0]<dim[0] and point[1]<dim[1]:
                ellvals.append(img.item(point))

        Barr[i][0] = statistics.median(ellvals)
    
    Sarr=[[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if j==0:
                Sarr[i][j]=1
            else:
                Sarr[i][j] = (1+i)*Sarr[i][j-1]
    S = np.array(Sarr)

    Carr=[[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if j>i:
                Carr[i][j] = 0
            elif j==0:
                if i==0:
                    Carr[i][j] =1
                else:
                    Carr[i][j] =-Carr[i-1][j]
            else:
                Carr[i][j] =-Carr[i][j-1]*(i-j+1)/j
    C = np.array(Carr)
    B = np.array(Barr)
    T = np.matmul(C,S)
    CB = np.matmul(C,B)

    for i in range(n-1, -1, -1):
        a[i] = CB.item((i,0))
        for j in range(i+1, n):
            a[i]-=(T.item((i,j))*a[j]*pow(samp2[j],j))
        a[i]/=(T.item(i,i)*pow(samp2[i],i))

    for x in range(dim[0]):
        for y in range(dim[1]):
            s2 = (x-c[0])**2 +  (ar*(y-c[1]))**2
            bg = 0
            for m in range(n,0,-1):
                bg*=s2
                bg+-a[m-1]
            flat1arr[x][y] = bg
            flat0arr[x][y] = img.item((x,y))-bg
    
    flat[0] = np.array(flat0arr)
    flat[1] = np.array(flat1arr)

def ellipsePoints(c, r):
    ellipse = []
    x=0
    y = r[1]
    dx,dy,d1,d2=0,0,0,0

    d1 = (r[1]*r[1]) - (r[0]*r[0]*r[1]) + (0.25*r[0]*r[0])
    dx = 2*r[1]*r[1]*x
    dy = 2*r[0]*r[0]*y
    
    while dx<dy:
        v1 = (int( x+c[0]),int( y+c[1]))
        v2 = (int(-x+c[0]),int( y+c[1]))
        v3 = (int( x+c[0]),int(-y+c[1]))
        v4 = (int(-x+c[0]),int(-y+c[1]))

        ellipse.append(v1)
        ellipse.append(v2)
        ellipse.append(v3)
        ellipse.append(v4)

        if d1<0:
            x+=1
            dx+= (2*r[1]*r[1])
            d1+= (dx+(r[1]*r[1]))
        else:
            x+=1
            y-=1
            dx+= (2*r[1]*r[1])
            d1 = d1 + dx - dy + (r[1]*r[1])
    
    d2 = ( (r[1]*r[1])*((x+0.5)*(x+0.5)) )\
            +( (r[0]*r[0]) * ((y-1)*(y-1)) )\
            -( r[0]*r[0]*r[1]*r[1])
    while y>=0:
        v1 = (int( x+c[0]), int( y+c[1]))
        v2 = (int(-x+c[0]), int( y+c[1]))
        v3 = (int( x+c[0]), int(-y+c[1]))
        v4 = (int(-x+c[0]), int(-y+c[1]))

        ellipse.append(v1)
        ellipse.append(v2)
        ellipse.append(v3)
        ellipse.append(v4)

        if d2>0:
            y-=1
            dy-= (2*r[0]*r[0])
            d2 = d2 + (r[0]*r[0]) - dy
        else:
            y-=1
            x+=1
            dx += (2*r[1]*r[1])
            dy -= (2*r[0]*r[0])
            d2 = d2 + dx - dy + (r[0]*r[0])

    return ellipse
