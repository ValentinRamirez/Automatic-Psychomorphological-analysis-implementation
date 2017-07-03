import cv2
import numpy as np
import matplotlib.pyplot as plt
import forehead_region_growing



def Exp_Ret(img,chin_area,lmarks):

    A=lmarks[0][8,:]
    B=lmarks[0][33,:]
    C=lmarks[0][14,:]
    D=lmarks[0][2,:]
    height=A[1]-C[1]
    Base=C[0]-D[0]
    #base=lmarks[0][7, 0]-lmarks[0][9, 0]
    #Trap_area=(Base+base)*hight/2
    Sq_area=Base*height
    Tr_area=Sq_area/2
    ER_ratio=(chin_area-Tr_area)/(Sq_area-Tr_area)
    plt.title('Expansion/Retraction')
    plt.imshow(img[:, :, ::-1])
    plt.scatter(C[0],C[1],c='r')
    plt.scatter(D[0],D[1],c='r')
    plt.scatter(A[0],A[1],c='r')
    plt.scatter(C[0],A[1],c='r')
    plt.scatter(D[0],A[1],c='r')
    plt.plot((C[0],D[0]),(C[1],D[1]),'k')
    plt.plot((C[0],A[0]),(C[1],A[1]),'k')
    plt.plot((D[0],A[0]),(D[1],A[1]),'k')
    plt.plot((C[0],C[0]),(C[1],A[1]),'k')
    plt.plot((D[0],D[0]),(D[1],A[1]),'k')
    plt.plot((C[0],D[0]),(A[1],A[1]),'k')
    plt.axis([0, img.shape[1],img.shape[0],0])
    plt.show()
    return 26.1474251256*ER_ratio-6.8793108487


def Triangle_of_senses(img,BF_area,lmarks):
    Base=lmarks[0][26, 0]-lmarks[0][17, 0]
    base=lmarks[0][54, 0]-lmarks[0][48, 0]
    top_eyebrow=min(lmarks[0][19, 1],lmarks[0][24, 1])
    height=lmarks[0][57, 1]-top_eyebrow
    lf_area=(Base+base)*height/2
    plt.figure()
    plt.title('Triangle of senses')
    plt.imshow(img[:, :, ::-1])
    plt.scatter(lmarks[0][26, 0],top_eyebrow,c='r')
    plt.scatter(lmarks[0][17, 0],top_eyebrow,c='r')
    plt.scatter(lmarks[0][54, 0],lmarks[0][57, 1],c='r')
    plt.scatter(lmarks[0][48, 0],lmarks[0][57, 1],c='r')
    plt.plot((lmarks[0][26, 0],lmarks[0][17, 0]),(top_eyebrow,top_eyebrow),'k')
    plt.plot((lmarks[0][26, 0],lmarks[0][54, 0]),(top_eyebrow,lmarks[0][57, 1]),'k')
    plt.plot((lmarks[0][54, 0],lmarks[0][48, 0]),(lmarks[0][57, 1],lmarks[0][57, 1]),'k')
    plt.plot((lmarks[0][48, 0],lmarks[0][17, 0]),(lmarks[0][57, 1],top_eyebrow),'k')
    plt.axis([0, img.shape[1],img.shape[0],0])
    plt.show()

    print '\n\nBig face area',BF_area
    print '\nLittle face area',lf_area
    return 26.4645944604*lf_area/BF_area-5.02233664555

def Three_zones(img,lmarks):
#Converting the image to Gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#SEEDS for posterior growing and edge calcuation
    #Forehead
    forehead_seed=(lmarks[0][17,:]+lmarks[0][26,:])/2
    forehead_seed=forehead_seed.astype(int)
    forehead_edge=forehead_seed[1]

#LEFT & RIGHT edges of the face
    left=lmarks[0][17,0]
    #left=(lmarks[0][0, 0]+lmarks[0][1, 0])/2
    left=left.astype(int)
    #right=lmarks[0][16, 0]
    right=lmarks[0][26,0]
    right=right.astype(int)

# RATIONAL ZONE:Region growing for the forehead with the previously found seed on the points above the inferior forehead limit.
    forehead_threshold = 60
    rational_img, rational_area = forehead_region_growing.forehead_region_growing(gray_img,forehead_seed,forehead_edge,0,left,right,forehead_threshold)
    plt.figure()
    plt.title('Rational zone')
    plt.imshow(rational_img[:, :])


# EMOTIONAL ZONE:
    BE1=lmarks[0][16, 0]-lmarks[0][0, 0]
    BE2=lmarks[0][15, 0]-lmarks[0][1, 0]
    BE3=lmarks[0][14, 0]-lmarks[0][2, 0]
    BE4=lmarks[0][13, 0]-lmarks[0][3, 0]
    HE1=lmarks[0][0, 1]-forehead_edge
    HE2=lmarks[0][1, 1]-lmarks[0][0, 1]
    HE3=lmarks[0][2, 1]-lmarks[0][1, 1]
    HE4=lmarks[0][33, 1]-lmarks[0][2, 1]
    Area_E=BE1*HE1+(BE1+BE2)*HE2/2+(BE2+BE3)*HE3/2+BE4*HE4

# INSTINCTIVE ZONE:
    #Region growing for the forehead with the previously found seed on the points
    #above the inferior forehead limit.
    BI1=BE4
    BI2=lmarks[0][12, 0]-lmarks[0][4, 0]
    BI3=lmarks[0][11, 0]-lmarks[0][5, 0]
    BI4=lmarks[0][10, 0]-lmarks[0][6, 0]
    BI5=lmarks[0][9, 0]-lmarks[0][7, 0]
    HI1=lmarks[0][3, 1]-lmarks[0][33, 1]
    HI2=lmarks[0][4, 1]-lmarks[0][3, 1]
    HI3=lmarks[0][5, 1]-lmarks[0][4, 1]
    HI4=lmarks[0][6, 1]-lmarks[0][5, 1]
    HI5=lmarks[0][7, 1]-lmarks[0][6, 1]
    HI6=lmarks[0][8, 1]-lmarks[0][7, 1]
    Chin_area=0.5*(BI5*HI6+(BI5+BI4)*HI5+(BI4+BI3)*HI4+(BI3+BI2)*HI3+(BI2+BI1)*HI2+(BI1+BE3)*(HE4+HI1))
    Area_I=0.5*(BI5*HI6+(BI5+BI4)*HI5+(BI4+BI3)*HI4+(BI3+BI2)*HI3+(BI2+BI1)*HI2)+BI1*HI1



    return rational_area,Area_E,Area_I,Chin_area
