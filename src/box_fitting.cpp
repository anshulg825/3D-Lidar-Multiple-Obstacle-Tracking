//
// Created by kosuke on 11/29/17.
//
#include <array>
#include <random>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include "box_fitting.h"

using namespace std;
using namespace pcl;
using namespace cv;

// float picScale = 10;
// float roiM = 90;
float picScale = 900/roiM;
int ramPoints = 120;
float lSlopeDist = 1.2;
int lnumPoints = 5;
float sensorHeight = 2.3;
// float tHeightMin = 1.2;
float tHeightMin = -2.00;
float tHeightMax = 3.00;
// float tWidthMin = 0.5;
// float tWidthMin = 0.4;
float tWidthMin = 0.25;
float tWidthMax = 4.5;
float tLenMin = 0.1;
float tLenMax = 20.0;
float tAreaMax = 4.0;
float tRatioMin = 0.5;
float tRatioMax = 4.5;
float minLenRatio = 0.5;
float tPtPerM3 = 8;

void getClusteredPoints(PointCloud<PointXYZ>::Ptr elevatedCloud,
                        array<array<int, numGrid>, numGrid> cartesianData,
                        vector<PointCloud<PointXYZ>>&  clusteredPoints) {
	// cout<<"getting clustered Points wait bitch"<<endl;
    for (int i = 0; i < elevatedCloud->size(); i++) {
        // cout<<"Mysterious ROIm  "<<roiM<<endl;
        float x = elevatedCloud->points[i].x;
        float y = elevatedCloud->points[i].y;
        float z = elevatedCloud->points[i].z;
        float xC = x + roiM / 2;
        float yC = y + roiM / 2;
        // exclude outside roi points
        if (xC < 0 || xC >= roiM || yC < 0 || yC >= roiM) continue;
        int xI = floor(numGrid * xC / roiM);
        int yI = floor(numGrid * yC / roiM);

        int clusterNum = cartesianData[xI][yI]; //1 ~ numCluster
        int vectorInd = clusterNum - 1; //0 ~ (numCluster -1 )
        if (clusterNum != 0) {
            PointXYZ o;
            o.x = x;
            o.y = y;
            o.z = z;
            clusteredPoints[vectorInd].push_back(o);
        }
    }
}

void getPointsInPcFrame(Point2f rectPoints[], vector<Point2f>& pcPoints, int offsetX, int offsetY){
    // loop 4 rect points
    for (int pointI = 0; pointI < 4; pointI++){
        float picX = rectPoints[pointI].x;
        float picY = rectPoints[pointI].y;
        // reverse offset
        float rOffsetX = picX - offsetX;
        float rOffsetY = picY - offsetY;
        // reverse from image coordinate to eucledian coordinate
        float rX = rOffsetX;
        float rY = picScale*roiM - rOffsetY;
        // reverse to 30mx30m scale
        float rmX = rX/picScale;
        float rmY = rY/picScale;
        // reverse from (0 < x,y < 30) to (-15 < x,y < 15)
        float pcX = rmX - roiM/2;
        float pcY = rmY - roiM/2;
        Point2f point(pcX, pcY);
        pcPoints[pointI] = point;
    }
}

bool ruleBasedFilter(vector<Point2f> pcPoints, float maxZ, int numPoints){
    bool isPromising = false;
    //minnimam points thresh
    if(numPoints < 60) return isPromising;
    // length is longest side of the rectangle while width is the shorter side.
    float width, length, height, area, ratio, mass;

    float x1 = pcPoints[0].x;
    float y1 = pcPoints[0].y;
    float x2 = pcPoints[1].x;
    float y2 = pcPoints[1].y;
    float x3 = pcPoints[2].x;
    float y3 = pcPoints[2].y;

    float dist1 = sqrt((x1-x2)*(x1-x2)+ (y1-y2)*(y1-y2));
    float dist2 = sqrt((x3-x2)*(x3-x2)+ (y3-y2)*(y3-y2));
    if(dist1 > dist2){
        length = dist1;
        width = dist2;
    }
    else{
        length = dist2;
        width = dist1;
    }
    // assuming ground = sensor height
    height = maxZ /* + sensorHeight*/;
    // assuming right angle
    area = dist1*dist2;
    mass = area*height;
    ratio = length/width;

    //start rule based filtering
    if(height > tHeightMin && height < tHeightMax)
    {
        if(width > tWidthMin && width < tWidthMax){
    //     	cout<<"width criteria satisfies"<<endl;
            if(length > tLenMin && length < tLenMax){
            	cout<<"Length criteria satisfied"<<endl;
                if(area < tAreaMax){
    //             	cout<<"Area criteria satisfied"<<endl;
                    if(numPoints > mass*tPtPerM3){
                    	cout<<"Density criteria satisfied"<<endl;
                        if(length > minLenRatio){
                        	// cout<<"Length ratio satisfied"<<endl;
                            if(ratio > tRatioMin && ratio < tRatioMax)
                            {
                            	cout<<"Everything is good"<<endl;
                                isPromising = true;
    //                             return isPromising;
                            }
                        }
    //                     else{
    //                         isPromising = true;
    //                         return isPromising;
                        }
                    }
                }
            }
    //     }
    }
    else
    {
        isPromising = false;
    }
    return isPromising;
}

void getBoundingBox(vector<PointCloud<PointXYZ>>  clusteredPoints,
                    vector<PointCloud<PointXYZ>>& bbPoints){
	// cout<<":getting bounding box wait bitch"<<endl;
    bbPoints.clear();
    for (int iCluster = 0; iCluster < clusteredPoints.size(); iCluster++)
    {
        // Mat m (picScale*roiM, picScale*roiM, CV_8UC1, Scalar(0));
        float initPX = clusteredPoints[iCluster][0].x + roiM/2;
        float initPY = clusteredPoints[iCluster][0].y + roiM/2;
        int initX = floor(initPX*picScale);
        int initY = floor(initPY*picScale);
        int initPicX = initX;
        int initPicY = picScale*roiM - initY;
        int offsetInitX = roiM*picScale/2 - initPicX;
        int offsetInitY = roiM*picScale/2 - initPicY;
        int numPoints = clusteredPoints[iCluster].size();
        vector<Point> pointVec(numPoints);
        vector<Point2f> pcPoints(4);
        float minMx, minMy, maxMx, maxMy;
        float minM = 999; float maxM = -999; float maxZ = -99;
        // for center of gravity
        float sumX = 0; float sumY = 0;
        // cout<<"Where have you been"<<" "<<clusteredPoints[iCluster].size()<<endl;
        for (int iPoint = 0; iPoint < clusteredPoints[iCluster].size(); iPoint++){
            float pX = clusteredPoints[iCluster][iPoint].x;
            float pY = clusteredPoints[iCluster][iPoint].y;
            float pZ = clusteredPoints[iCluster][iPoint].z;
            // cast (-15 < x,y < 15) into (0 < x,y < 30)
            float roiX = pX + roiM/2;
            float roiY = pY + roiM/2;
            // cast 30mx30m into 900x900 scale
            int x = floor(roiX*picScale);
            int y = floor(roiY*picScale);
            // cast into image coordinate
            int picX = x;
            int picY = picScale*roiM - y;
            // offset so that the object would be locate at the center
            int offsetX = picX + offsetInitX;
            int offsetY = picY + offsetInitY;
            // m.at<uchar>(offsetY, offsetX) = 255;
            pointVec[iPoint] = Point(offsetX, offsetY);
            // calculate min and max slope
            float m_ = pY/pX;
            // cout<<"Printing random values to check "<<m_<<endl;
            if(m_ < minM) {
                minM = m_;
                minMx = pX;
                minMy = pY;
            }
            if(m_ > maxM) {
                maxM = m_;
                maxMx = pX;
                maxMy = pY;
            }

            //get maxZ
            // cout<<"********************PZ********************   "<<pZ<<endl;
            if(pZ > maxZ) 
                {   
                    // cout<<"IN if trueee"<<endl;
                    maxZ = pZ;
                }
                // cout<<"---------------------------MAXZ-------------------------  "<<maxZ<<endl;
            sumX += offsetX;
            sumY += offsetY; 

        }
        // cout<<"Shine bright like a diamond line 198 box fitting"<<endl;
        // L shape fitting parameters
        float xDist = maxMx - minMx;
        float yDist = maxMy - minMy;
        float slopeDist = sqrt(xDist*xDist + yDist*yDist);
        float slope = (maxMy - minMy)/(maxMx - minMx);

        // random variable
        mt19937_64 mt(0);
        uniform_int_distribution<> randPoints(0, numPoints-1);
        // cout<<"Life is random. So random"<<endl;
        // start l shape fitting for car like object
        // lSlopeDist = 30, lnumPoints = 300
        
        // cout<<"Printing some values"<< slopeDist<< " "<<lSlopeDist<<" "<<numPoints<< " "<<lnumPoints<<endl;
        if(slopeDist > lSlopeDist && numPoints > lnumPoints){
        	// cout<<"*****************************I don't have enough points**********************************"<<endl;
            float maxDist = 0;
            float maxDx, maxDy;

            // 80 random points, get max distance
            for(int i = 0; i < ramPoints; i++){

                int pInd = randPoints(mt);
            	// int pInd= 0;
                // assert(pInd >= 0 && pInd < clusteredPoints[iCluster].size());
                
                float xI = clusteredPoints[iCluster][pInd].x;
                float yI = clusteredPoints[iCluster][pInd].y;

                // from equation of distance between line and point
                float dist = abs(slope*xI-1*yI+maxMy-slope*maxMx)/sqrt(slope*slope + 1);
                if(dist > maxDist) {
                    maxDist = dist;
                    maxDx = xI;
                    maxDy = yI;
                }
            }
            
            // for center of gravity
            // maxDx = sumX/clusteredPoints[iCluster].size();
            // maxDy = sumY/clusteredPoints[iCluster].size();

            // vector adding
            float maxMvecX = maxMx - maxDx;
            float maxMvecY = maxMy - maxDy;
            float minMvecX = minMx - maxDx;
            float minMvecY = minMy - maxDy;
            float lastX = maxDx + maxMvecX + minMvecX;
            float lastY = maxDy + maxMvecY + minMvecY;

            pcPoints[0] = Point2f(minMx, minMy);
            pcPoints[1] = Point2f(maxDx, maxDy);
            pcPoints[2] = Point2f(maxMx, maxMy);
            pcPoints[3] = Point2f(lastX, lastY);
            bool isPromising = ruleBasedFilter(pcPoints, maxZ, numPoints);
            if(!isPromising) continue;
        }
        else{
            //MAR fitting
        	// cout<<"STARTING MY JOURNEY"<<endl;
        	// cout<<"I am feeling sleepy"<<pointVec.size()<<endl;
            RotatedRect rectInfo = minAreaRect(pointVec);
            // cout<<"PLease spare me GOD"<<endl;
            Point2f rectPoints[4]; rectInfo.points( rectPoints );
            // covert points back to lidar coordinate
             // cout<<"Ohh na na what's my name"<< endl;
            getPointsInPcFrame(rectPoints, pcPoints, offsetInitX, offsetInitY);
            // rule based filter
            // cout<<"It'll kill you. ONly if I die"<<endl;
            bool isPromising = ruleBasedFilter(pcPoints, maxZ, numPoints);
            // cout<<"work work work work work work"<<endl;
            if(!isPromising) 
            	{
            continue;
        }
            // for visualization
        }

        // make pcl cloud for 3d bounding box
        // cout<<"This is what you came for"<<" "<<pcPoints.size()<<endl;
        PointCloud<PointXYZ> oneBbox;
        for(int pclH = 0; pclH < 2; pclH++){
            for(int pclP = 0; pclP < 4; pclP++){
                PointXYZ o;
                o.x = pcPoints[pclP].x;
                o.y = pcPoints[pclP].y;
                if(pclH == 0) o.z = -sensorHeight;
                else o.z = maxZ;
                oneBbox.push_back(o);
            }
        }
        bbPoints.push_back(oneBbox);

//        clustered2D[iCluster] = m;
    }
    // cout<<"done getting bboxes, Now bring me Thanos"<<endl;
}

vector<PointCloud<PointXYZ>> boxFitting(PointCloud<PointXYZ>::Ptr elevatedCloud,
                array<array<int, numGrid>, numGrid> cartesianData,
                int numCluster){
    vector<PointCloud<PointXYZ>>  clusteredPoints(numCluster);
    getClusteredPoints(elevatedCloud, cartesianData, clusteredPoints);
    cout<<"clusteredPoints in box fitting "<< clusteredPoints.size()<<endl;
    vector<PointCloud<PointXYZ>>  bbPoints;
    getBoundingBox(clusteredPoints, bbPoints);
    // cout<<"All boxes fit. Thunder feel the Thunder"<<endl;
    return bbPoints;
//    vector<vector<float>>  bBoxes(numCluster,  vector<float>(6));
//
//    return bBoxes;
}