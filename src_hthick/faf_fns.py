import cv2 as cv
import numpy as np

def ellipsePoints(c, r):
    ellipse = []
    x=0
    y = r[1]
    dx,dy,d1,d2=0,0,0,0

    d1 = (r[1]*r[1]) - (r[0]*r[0]*r[1]) + (0.25*r[0]*r[0])
    dx = 2*r[1]*r[1]*x
    dy = 2*r[0]*r[0]*y
    
    ellipse.append(( x+c[0], y+c[1]))
    ellipse.append((-x+c[0], y+c[1]))
    ellipse.append(( x+c[0],-y+c[1]))
    ellipse.append((-x+c[0],-y+c[1]))

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
    while y>0:
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
