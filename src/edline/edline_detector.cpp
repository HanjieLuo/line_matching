#include "edline/edline_detector.h"

#include <sys/time.h>

#define Horizontal 255  //if |dx|<|dy|;
#define Vertical 0      //if |dy|<=|dx|;
#define UpDir 1
#define RightDir 2
#define DownDir 3
#define LeftDir 4
#define TryTime 6
#define SkipEdgePoint 2

//#define DEBUGEdgeDrawing
//#define DEBUGEDLine

using namespace std;
EDLineDetector::EDLineDetector() {
    //	cout<<"Call EDLineDetector constructor function"<<endl;
    //set parameters for line segment detection
    ksize_ = 5;
    sigma_ = 1.0;
    gradienThreshold_ = 80;  // ***** ORIGINAL WAS 25
    anchorThreshold_ = 2;    //8
    scanIntervals_ = 2;      //2
    minLineLen_ = 15;
    lineFitErrThreshold_ = 1.4;
    InitEDLine_();
}
EDLineDetector::EDLineDetector(EDLineParam param) {
    //set parameters for line segment detection
    ksize_ = param.ksize;
    sigma_ = param.sigma;
    gradienThreshold_ = param.gradientThreshold;
    anchorThreshold_ = param.anchorThreshold;
    scanIntervals_ = param.scanIntervals;
    minLineLen_ = param.minLineLen;
    lineFitErrThreshold_ = param.lineFitErrThreshold;
    InitEDLine_();
}
void EDLineDetector::InitEDLine_() {
    dxImg_.create(1, 1, CV_16SC1);
    dyImg_.create(1, 1, CV_16SC1);
    gImgWO_.create(1, 1, CV_8SC1);
    pFirstPartEdgeX_ = NULL;
    pFirstPartEdgeY_ = NULL;
    pFirstPartEdgeS_ = NULL;
    pSecondPartEdgeX_ = NULL;
    pSecondPartEdgeY_ = NULL;
    pSecondPartEdgeS_ = NULL;
    pAnchorX_ = NULL;
    pAnchorY_ = NULL;
}
EDLineDetector::~EDLineDetector() {
    if (pFirstPartEdgeX_ != NULL) {
        delete[] pFirstPartEdgeX_;
        delete[] pFirstPartEdgeY_;
        delete[] pSecondPartEdgeX_;
        delete[] pSecondPartEdgeY_;
        delete[] pAnchorX_;
        delete[] pAnchorY_;
    }
    if (pFirstPartEdgeS_ != NULL) {
        delete[] pFirstPartEdgeS_;
        delete[] pSecondPartEdgeS_;
    }
}

void writeMat(cv::Mat m, string name, int n) {
    std::stringstream ss;
    string s;
    ss << n;
    ss >> s;
    string fileNameConf = name + s;
    cv::FileStorage fsConf(fileNameConf, cv::FileStorage::WRITE);
    fsConf << "m" << m;

    fsConf.release();
}

int EDLineDetector::EdgeDrawing(cv::Mat &image, EdgeChains &edgeChains, bool smoothed) {
    if (!smoothed) {  //input image hasn't been smoothed.
        cv::GaussianBlur(image, image_, cv::Size(ksize_, ksize_), sigma_);
    } else {
        image_ = image;
    }

    imageWidth = image_.cols;
    imageHeight = image_.rows;
    unsigned int pixelNum = imageWidth * imageHeight;

    unsigned int edgePixelArraySize = pixelNum / 5;
    unsigned int maxNumOfEdge = edgePixelArraySize / 20;
    //compute dx, dy images
    if (gImg_.cols != (int)imageWidth || gImg_.rows != (int)imageHeight) {
        if (pFirstPartEdgeX_ != NULL) {
            delete[] pFirstPartEdgeX_;
            delete[] pFirstPartEdgeY_;
            delete[] pSecondPartEdgeX_;
            delete[] pSecondPartEdgeY_;
            delete[] pFirstPartEdgeS_;
            delete[] pSecondPartEdgeS_;
            delete[] pAnchorX_;
            delete[] pAnchorY_;
        }

        dxImg_.create(imageHeight, imageWidth, CV_16SC1);
        dyImg_.create(imageHeight, imageWidth, CV_16SC1);
        gImgWO_.create(imageHeight, imageWidth, CV_16SC1);
        gImg_.create(imageHeight, imageWidth, CV_16SC1);
        dirImg_.create(imageHeight, imageWidth, CV_8UC1);
        edgeImage_.create(imageHeight, imageWidth, CV_8UC1);
        pFirstPartEdgeX_ = new unsigned int[edgePixelArraySize];
        pFirstPartEdgeY_ = new unsigned int[edgePixelArraySize];
        pSecondPartEdgeX_ = new unsigned int[edgePixelArraySize];
        pSecondPartEdgeY_ = new unsigned int[edgePixelArraySize];
        pFirstPartEdgeS_ = new unsigned int[maxNumOfEdge];
        pSecondPartEdgeS_ = new unsigned int[maxNumOfEdge];
        pAnchorX_ = new unsigned int[edgePixelArraySize];
        pAnchorY_ = new unsigned int[edgePixelArraySize];

        zeroMat_ = cv::Mat::zeros(imageHeight, imageWidth, CV_16SC1);
    }

    cv::Sobel(image_, dxImg_, CV_16SC1, 1, 0, 3);
    cv::Sobel(image_, dyImg_, CV_16SC1, 0, 1, 3);

    cv::absdiff(dxImg_, zeroMat_, dxABS_m_);
    cv::absdiff(dyImg_, zeroMat_, dyABS_m_);

    cv::add(dyABS_m_, dxABS_m_, sumDxDy_);

    cv::threshold(sumDxDy_, gImg_, gradienThreshold_ + 1, 255, cv::THRESH_TOZERO);
    gImg_ = gImg_ / 4;
    gImgWO_ = sumDxDy_ / 4;
    cv::compare(dxABS_m_, dyABS_m_, dirImg_, cv::CMP_LT);

    short *pgImg = gImg_.ptr<short>();
    unsigned char *pdirImg = dirImg_.ptr();

    //extract the anchors in the gradient image, store into a vector
    memset(pAnchorX_, 0, edgePixelArraySize * sizeof(unsigned int));  //initialization
    memset(pAnchorY_, 0, edgePixelArraySize * sizeof(unsigned int));
    unsigned int anchorsSize = 0;
    int indexInArray;
    unsigned char gValue1, gValue2, gValue3;

    for (unsigned int w = 1; w < imageWidth - 1; w = w + scanIntervals_) {
        for (unsigned int h = 1; h < imageHeight - 1; h = h + scanIntervals_) {
            indexInArray = h * imageWidth + w;
            if (pdirImg[indexInArray] == Horizontal) {                                                                                                                           //if the direction of pixel is horizontal, then compare with up and down
                                                                                                                                                                                 //gValue2 = pgImg[indexInArray];
                if (pgImg[indexInArray] >= pgImg[indexInArray - imageWidth] + anchorThreshold_ && pgImg[indexInArray] >= pgImg[indexInArray + imageWidth] + anchorThreshold_) {  // (w,h) is accepted as an anchor
                    pAnchorX_[anchorsSize] = w;
                    pAnchorY_[anchorsSize++] = h;
                }
            } else {
                if (pgImg[indexInArray] >= pgImg[indexInArray - 1] + anchorThreshold_ && pgImg[indexInArray] >= pgImg[indexInArray + 1] + anchorThreshold_) {  // (w,h) is accepted as an anchor
                    pAnchorX_[anchorsSize] = w;
                    pAnchorY_[anchorsSize++] = h;
                }
            }
        }
    }

    if (anchorsSize > edgePixelArraySize) {
        std::cout << "anchor size is larger than its maximal size. anchorsSize=" << anchorsSize << ", maximal size = " << edgePixelArraySize << std::endl;
        return -1;
    }

    //link the anchors by smart routing
    edgeImage_.setTo(0);
    unsigned char *pEdgeImg = edgeImage_.data;
    memset(pFirstPartEdgeX_, 0, edgePixelArraySize * sizeof(unsigned int));  //initialization
    memset(pFirstPartEdgeY_, 0, edgePixelArraySize * sizeof(unsigned int));
    memset(pSecondPartEdgeX_, 0, edgePixelArraySize * sizeof(unsigned int));
    memset(pSecondPartEdgeY_, 0, edgePixelArraySize * sizeof(unsigned int));
    memset(pFirstPartEdgeS_, 0, maxNumOfEdge * sizeof(unsigned int));
    memset(pSecondPartEdgeS_, 0, maxNumOfEdge * sizeof(unsigned int));
    unsigned int offsetPFirst = 0, offsetPSecond = 0;
    unsigned int offsetPS = 0;

    unsigned int x, y;
    unsigned int lastX = 0;
    unsigned int lastY = 0;
    unsigned char lastDirection;      //up = 1, right = 2, down = 3, left = 4;
    unsigned char shouldGoDirection;  //up = 1, right = 2, down = 3, left = 4;
    int edgeLenFirst, edgeLenSecond;

    // gettimeofday(&t1, NULL);
    for (unsigned int i = 0; i < anchorsSize; i++) {
        x = pAnchorX_[i];
        y = pAnchorY_[i];
        indexInArray = y * imageWidth + x;
        if (pEdgeImg[indexInArray]) {  //if anchor i is already been an edge pixel.
            continue;
        }
        /*The walk stops under 3 conditions:
     * 1. We move out of the edge areas, i.e., the thresholded gradient value
     *    of the current pixel is 0.
     * 2. The current direction of the edge changes, i.e., from horizontal
     *    to vertical or vice versa.?? (This is turned out not correct. From the online edge draw demo
     *    we can figure out that authors don't implement this rule either because their extracted edge
     *    chain could be a circle which means pixel directions would definitely be different
     *    in somewhere on the chain.)
     * 3. We encounter a previously detected edge pixel. */
        pFirstPartEdgeS_[offsetPS] = offsetPFirst;
        if (pdirImg[indexInArray] == Horizontal) {  //if the direction of this pixel is horizontal, then go left and right.
            //fist go right, pixel direction may be different during linking.
            lastDirection = RightDir;
            while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray]) {
                pEdgeImg[indexInArray] = 1;  // Mark this pixel as an edge pixel
                pFirstPartEdgeX_[offsetPFirst] = x;
                pFirstPartEdgeY_[offsetPFirst++] = y;
                shouldGoDirection = 0;                                         //unknown
                if (pdirImg[indexInArray] == Horizontal) {                     //should go left or right
                    if (lastDirection == UpDir || lastDirection == DownDir) {  //change the pixel direction now
                        if (x > lastX) {                                       //should go right
                            shouldGoDirection = RightDir;
                        } else {  //should go left
                            shouldGoDirection = LeftDir;
                        }
                    }
                    lastX = x;
                    lastY = y;
                    if (lastDirection == RightDir || shouldGoDirection == RightDir) {  //go right
                        if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) {   //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the right and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray + 1];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth + 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-right
                            x = x + 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-right
                            x = x + 1;
                            y = y + 1;
                        } else {  //straight-right
                            x = x + 1;
                        }
                        lastDirection = RightDir;
                    } else if (lastDirection == LeftDir || shouldGoDirection == LeftDir) {  //go left
                        if (x == 0 || y == 0 || y == imageHeight - 1) {                     //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the left and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth - 1];
                        gValue2 = (unsigned char)pgImg[indexInArray - 1];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-left
                            x = x - 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-left
                            x = x - 1;
                            y = y + 1;
                        } else {  //straight-left
                            x = x - 1;
                        }
                        lastDirection = LeftDir;
                    }
                } else {                                                          //should go up or down.
                    if (lastDirection == RightDir || lastDirection == LeftDir) {  //change the pixel direction now
                        if (y > lastY) {                                          //should go down
                            shouldGoDirection = DownDir;
                        } else {  //should go up
                            shouldGoDirection = UpDir;
                        }
                    }
                    lastX = x;
                    lastY = y;
                    if (lastDirection == DownDir || shouldGoDirection == DownDir) {   //go down
                        if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) {  //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the down and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray + imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray + imageWidth];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //down-right
                            x = x + 1;
                            y = y + 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-left
                            x = x - 1;
                            y = y + 1;
                        } else {  //straight-down
                            y = y + 1;
                        }
                        lastDirection = DownDir;
                    } else if (lastDirection == UpDir || shouldGoDirection == UpDir) {  //go up
                        if (x == 0 || x == imageWidth - 1 || y == 0) {                  //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the up and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray - imageWidth];
                        gValue3 = (unsigned char)pgImg[indexInArray - imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-right
                            x = x + 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //up-left
                            x = x - 1;
                            y = y - 1;
                        } else {  //straight-up
                            y = y - 1;
                        }
                        lastDirection = UpDir;
                    }
                }
                indexInArray = y * imageWidth + x;
            }  //end while go right
               //then go left, pixel direction may be different during linking.
            x = pAnchorX_[i];
            y = pAnchorY_[i];
            indexInArray = y * imageWidth + x;
            pEdgeImg[indexInArray] = 0;  //mark the anchor point be a non-edge pixel and
            lastDirection = LeftDir;
            pSecondPartEdgeS_[offsetPS] = offsetPSecond;
            while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray]) {
                pEdgeImg[indexInArray] = 1;  // Mark this pixel as an edge pixel
                pSecondPartEdgeX_[offsetPSecond] = x;
                pSecondPartEdgeY_[offsetPSecond++] = y;
                shouldGoDirection = 0;                                         //unknown
                if (pdirImg[indexInArray] == Horizontal) {                     //should go left or right
                    if (lastDirection == UpDir || lastDirection == DownDir) {  //change the pixel direction now
                        if (x > lastX) {                                       //should go right
                            shouldGoDirection = RightDir;
                        } else {  //should go left
                            shouldGoDirection = LeftDir;
                        }
                    }
                    lastX = x;
                    lastY = y;
                    if (lastDirection == RightDir || shouldGoDirection == RightDir) {  //go right
                        if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) {   //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the right and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray + 1];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth + 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-right
                            x = x + 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-right
                            x = x + 1;
                            y = y + 1;
                        } else {  //straight-right
                            x = x + 1;
                        }
                        lastDirection = RightDir;
                    } else if (lastDirection == LeftDir || shouldGoDirection == LeftDir) {  //go left
                        if (x == 0 || y == 0 || y == imageHeight - 1) {                     //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the left and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth - 1];
                        gValue2 = (unsigned char)pgImg[indexInArray - 1];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-left
                            x = x - 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-left
                            x = x - 1;
                            y = y + 1;
                        } else {  //straight-left
                            x = x - 1;
                        }
                        lastDirection = LeftDir;
                    }
                } else {                                                          //should go up or down.
                    if (lastDirection == RightDir || lastDirection == LeftDir) {  //change the pixel direction now
                        if (y > lastY) {                                          //should go down
                            shouldGoDirection = DownDir;
                        } else {  //should go up
                            shouldGoDirection = UpDir;
                        }
                    }
                    lastX = x;
                    lastY = y;
                    if (lastDirection == DownDir || shouldGoDirection == DownDir) {   //go down
                        if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) {  //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the down and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray + imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray + imageWidth];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //down-right
                            x = x + 1;
                            y = y + 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-left
                            x = x - 1;
                            y = y + 1;
                        } else {  //straight-down
                            y = y + 1;
                        }
                        lastDirection = DownDir;
                    } else if (lastDirection == UpDir || shouldGoDirection == UpDir) {  //go up
                        if (x == 0 || x == imageWidth - 1 || y == 0) {                  //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the up and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray - imageWidth];
                        gValue3 = (unsigned char)pgImg[indexInArray - imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-right
                            x = x + 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //up-left
                            x = x - 1;
                            y = y - 1;
                        } else {  //straight-up
                            y = y - 1;
                        }
                        lastDirection = UpDir;
                    }
                }
                indexInArray = y * imageWidth + x;
            }     //end while go left
                  //end anchor is Horizontal
        } else {  //the direction of this pixel is vertical, go up and down
            //fist go down, pixel direction may be different during linking.
            lastDirection = DownDir;
            while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray]) {
                pEdgeImg[indexInArray] = 1;  // Mark this pixel as an edge pixel
                pFirstPartEdgeX_[offsetPFirst] = x;
                pFirstPartEdgeY_[offsetPFirst++] = y;
                shouldGoDirection = 0;                                         //unknown
                if (pdirImg[indexInArray] == Horizontal) {                     //should go left or right
                    if (lastDirection == UpDir || lastDirection == DownDir) {  //change the pixel direction now
                        if (x > lastX) {                                       //should go right
                            shouldGoDirection = RightDir;
                        } else {  //should go left
                            shouldGoDirection = LeftDir;
                        }
                    }
                    lastX = x;
                    lastY = y;
                    if (lastDirection == RightDir || shouldGoDirection == RightDir) {  //go right
                        if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) {   //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the right and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray + 1];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth + 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-right
                            x = x + 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-right
                            x = x + 1;
                            y = y + 1;
                        } else {  //straight-right
                            x = x + 1;
                        }
                        lastDirection = RightDir;
                    } else if (lastDirection == LeftDir || shouldGoDirection == LeftDir) {  //go left
                        if (x == 0 || y == 0 || y == imageHeight - 1) {                     //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the left and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth - 1];
                        gValue2 = (unsigned char)pgImg[indexInArray - 1];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-left
                            x = x - 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-left
                            x = x - 1;
                            y = y + 1;
                        } else {  //straight-left
                            x = x - 1;
                        }
                        lastDirection = LeftDir;
                    }
                } else {                                                          //should go up or down.
                    if (lastDirection == RightDir || lastDirection == LeftDir) {  //change the pixel direction now
                        if (y > lastY) {                                          //should go down
                            shouldGoDirection = DownDir;
                        } else {  //should go up
                            shouldGoDirection = UpDir;
                        }
                    }
                    lastX = x;
                    lastY = y;
                    if (lastDirection == DownDir || shouldGoDirection == DownDir) {   //go down
                        if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) {  //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the down and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray + imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray + imageWidth];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //down-right
                            x = x + 1;
                            y = y + 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-left
                            x = x - 1;
                            y = y + 1;
                        } else {  //straight-down
                            y = y + 1;
                        }
                        lastDirection = DownDir;
                    } else if (lastDirection == UpDir || shouldGoDirection == UpDir) {  //go up
                        if (x == 0 || x == imageWidth - 1 || y == 0) {                  //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the up and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray - imageWidth];
                        gValue3 = (unsigned char)pgImg[indexInArray - imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-right
                            x = x + 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //up-left
                            x = x - 1;
                            y = y - 1;
                        } else {  //straight-up
                            y = y - 1;
                        }
                        lastDirection = UpDir;
                    }
                }
                indexInArray = y * imageWidth + x;
            }  //end while go down
               //then go up, pixel direction may be different during linking.
            lastDirection = UpDir;
            x = pAnchorX_[i];
            y = pAnchorY_[i];
            indexInArray = y * imageWidth + x;
            pEdgeImg[indexInArray] = 0;  //mark the anchor point be a non-edge pixel and
            pSecondPartEdgeS_[offsetPS] = offsetPSecond;
            while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray]) {
                pEdgeImg[indexInArray] = 1;  // Mark this pixel as an edge pixel
                pSecondPartEdgeX_[offsetPSecond] = x;
                pSecondPartEdgeY_[offsetPSecond++] = y;
                shouldGoDirection = 0;                                         //unknown
                if (pdirImg[indexInArray] == Horizontal) {                     //should go left or right
                    if (lastDirection == UpDir || lastDirection == DownDir) {  //change the pixel direction now
                        if (x > lastX) {                                       //should go right
                            shouldGoDirection = RightDir;
                        } else {  //should go left
                            shouldGoDirection = LeftDir;
                        }
                    }
                    lastX = x;
                    lastY = y;
                    if (lastDirection == RightDir || shouldGoDirection == RightDir) {  //go right
                        if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) {   //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the right and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray + 1];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth + 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-right
                            x = x + 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-right
                            x = x + 1;
                            y = y + 1;
                        } else {  //straight-right
                            x = x + 1;
                        }
                        lastDirection = RightDir;
                    } else if (lastDirection == LeftDir || shouldGoDirection == LeftDir) {  //go left
                        if (x == 0 || y == 0 || y == imageHeight - 1) {                     //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the left and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth - 1];
                        gValue2 = (unsigned char)pgImg[indexInArray - 1];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-left
                            x = x - 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-left
                            x = x - 1;
                            y = y + 1;
                        } else {  //straight-left
                            x = x - 1;
                        }
                        lastDirection = LeftDir;
                    }
                } else {                                                          //should go up or down.
                    if (lastDirection == RightDir || lastDirection == LeftDir) {  //change the pixel direction now
                        if (y > lastY) {                                          //should go down
                            shouldGoDirection = DownDir;
                        } else {  //should go up
                            shouldGoDirection = UpDir;
                        }
                    }
                    lastX = x;
                    lastY = y;
                    if (lastDirection == DownDir || shouldGoDirection == DownDir) {   //go down
                        if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) {  //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the down and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray + imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray + imageWidth];
                        gValue3 = (unsigned char)pgImg[indexInArray + imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //down-right
                            x = x + 1;
                            y = y + 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //down-left
                            x = x - 1;
                            y = y + 1;
                        } else {  //straight-down
                            y = y + 1;
                        }
                        lastDirection = DownDir;
                    } else if (lastDirection == UpDir || shouldGoDirection == UpDir) {  //go up
                        if (x == 0 || x == imageWidth - 1 || y == 0) {                  //reach the image border
                            break;
                        }
                        // Look at 3 neighbors to the up and pick the one with the max. gradient value
                        gValue1 = (unsigned char)pgImg[indexInArray - imageWidth + 1];
                        gValue2 = (unsigned char)pgImg[indexInArray - imageWidth];
                        gValue3 = (unsigned char)pgImg[indexInArray - imageWidth - 1];
                        if (gValue1 >= gValue2 && gValue1 >= gValue3) {  //up-right
                            x = x + 1;
                            y = y - 1;
                        } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {  //up-left
                            x = x - 1;
                            y = y - 1;
                        } else {  //straight-up
                            y = y - 1;
                        }
                        lastDirection = UpDir;
                    }
                }
                indexInArray = y * imageWidth + x;
            }  //end while go up
        }      //end anchor is Vertical
               //only keep the edge chains whose length is larger than the minLineLen_;
        edgeLenFirst = offsetPFirst - pFirstPartEdgeS_[offsetPS];
        edgeLenSecond = offsetPSecond - pSecondPartEdgeS_[offsetPS];
        if (edgeLenFirst + edgeLenSecond < minLineLen_ + 1) {  //short edge, drop it
            offsetPFirst = pFirstPartEdgeS_[offsetPS];
            offsetPSecond = pSecondPartEdgeS_[offsetPS];
        } else {
            offsetPS++;
        }
    }

    // gettimeofday(&t2, NULL);
    // std::cout << "4:" << (t2.tv_sec - t1.tv_sec) * 1000. + (t2.tv_usec - t1.tv_usec) / 1000. << std::endl;

    //store the last index
    pFirstPartEdgeS_[offsetPS] = offsetPFirst;
    pSecondPartEdgeS_[offsetPS] = offsetPSecond;
    if (offsetPS > maxNumOfEdge) {
        std::cout << "Edge drawing Error: The total number of edges is larger than MaxNumOfEdge, "
                     "numofedge = "
                  << offsetPS << ", MaxNumOfEdge=" << maxNumOfEdge << std::endl;
        return -1;
    }
    if (offsetPFirst > edgePixelArraySize || offsetPSecond > edgePixelArraySize) {
        std::cout << "Edge drawing Error: The total number of edge pixels is larger than MaxNumOfEdgePixels, "
                     "numofedgePixel1 = "
                  << offsetPFirst << ",  numofedgePixel2 = " << offsetPSecond << ", MaxNumOfEdgePixel=" << edgePixelArraySize << std::endl;
        return -1;
    }
    if (!(offsetPFirst && offsetPSecond)) {
        std::cout << "Edge drawing Error: lines not found" << std::endl;
        return -1;
    }

    /*now all the edge information are stored in pFirstPartEdgeX_, pFirstPartEdgeY_,
   *pFirstPartEdgeS_,  pSecondPartEdgeX_, pSecondPartEdgeY_, pSecondPartEdgeS_;
   *we should reorganize them into edgeChains for easily using. */
    int tempID;
    edgeChains.xCors.resize(offsetPFirst + offsetPSecond);
    edgeChains.yCors.resize(offsetPFirst + offsetPSecond);
    edgeChains.sId.resize(offsetPS + 1);
    unsigned int *pxCors = &edgeChains.xCors.front();
    unsigned int *pyCors = &edgeChains.yCors.front();
    unsigned int *psId = &edgeChains.sId.front();
    offsetPFirst = 0;
    offsetPSecond = 0;
    unsigned int indexInCors = 0;
    unsigned int numOfEdges = 0;
    // gettimeofday(&t1, NULL);
    for (unsigned int edgeId = 0; edgeId < offsetPS; edgeId++) {
        //step1, put the first and second parts edge coordinates together from edge start to edge end
        psId[numOfEdges++] = indexInCors;
        indexInArray = pFirstPartEdgeS_[edgeId];
        offsetPFirst = pFirstPartEdgeS_[edgeId + 1];
        for (tempID = offsetPFirst - 1; tempID >= indexInArray; tempID--) {  //add first part edge
            pxCors[indexInCors] = pFirstPartEdgeX_[tempID];
            pyCors[indexInCors++] = pFirstPartEdgeY_[tempID];
        }
        indexInArray = pSecondPartEdgeS_[edgeId];
        offsetPSecond = pSecondPartEdgeS_[edgeId + 1];
        for (tempID = indexInArray + 1; tempID < (int)offsetPSecond; tempID++) {  //add second part edge
            pxCors[indexInCors] = pSecondPartEdgeX_[tempID];
            pyCors[indexInCors++] = pSecondPartEdgeY_[tempID];
        }
    }
    // gettimeofday(&t2, NULL);
    // std::cout << "5:" << (t2.tv_sec - t1.tv_sec) * 1000. + (t2.tv_usec - t1.tv_usec) / 1000. << std::endl;

    psId[numOfEdges] = indexInCors;  //the end index of the last edge
    edgeChains.numOfEdges = numOfEdges;

    return 1;
}

EDLineDetector::EDLineDetectorParallel::EDLineDetectorParallel(EDLineDetector *_edline,
                                                               EdgeChains *_edges,
                                                               double _logNT,
                                                               cv::Mutex &_lock,
                                                               std::vector<Line> *_lines) : edline(_edline), edges(_edges), logNT(_logNT), lock(_lock), lines(_lines) {
    minLineLen = edline->minLineLen_;
    lineFitErrThreshold = edline->lineFitErrThreshold_;
    imageWidth = edline->imageWidth;
    imageHeight = edline->imageHeight;

    dirImg = edline->dirImg_;
    dxImg = edline->dxImg_;  //store the dxImg;
    dyImg = edline->dyImg_;  //store the dyImg;

    bValidate = true;
}

double EDLineDetector::EDLineDetectorParallel::LeastSquaresLineFit(cv::Mat_<float> &ATA, cv::Mat_<float> &ATV, cv::Mat_<float> &fitMatT, cv::Mat_<float> &fitVec,
                                                                   unsigned int *xCors, unsigned int *yCors,
                                                                   unsigned int offsetS, std::array<double, 2> &lineEquation) const {
    if (lineEquation.size() != 2) {
        std::cout << "SHOULD NOT BE != 2" << std::endl;
        CV_Assert(false);
    }

    float *pMatT;
    float *pATA;
    double fitError = 0;
    double coef;
    unsigned char *pdirImg = dirImg.data;
    unsigned int offset = offsetS;
    /*If the first pixel in this chain is horizontal,
	 *then we try to find a horizontal line, y=ax+b;*/
    if (pdirImg[yCors[offsetS] * imageWidth + xCors[offsetS]] == Horizontal) {
        /*Build the system,and solve it using least square regression: mat * [a,b]^T = vec
		 * [x0,1]         [y0]
		 * [x1,1] [a]     [y1]
		 *    .   [b]  =   .
		 * [xn,1]         [yn]*/
        pMatT = fitMatT.ptr<float>();  //fitMatT = [x0, x1, ... xn; 1,1,...,1];
        for (int i = 0; i < minLineLen; i++) {
            //*(pMatT+minLineLen_) = 1; //the value are not changed;
            *(pMatT++) = xCors[offsetS];
            fitVec[0][i] = yCors[offsetS++];
        }
        ATA = fitMatT * fitMatT.t();
        ATV = fitMatT * fitVec.t();
        /* [a,b]^T = Inv(mat^T * mat) * mat^T * vec */
        pATA = ATA.ptr<float>();
        coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
        //		lineEquation = svd.Invert(ATA) * matT * vec;
        lineEquation[0] = coef * (double(pATA[3]) * double(ATV[0][0]) - double(pATA[1]) * double(ATV[0][1]));
        lineEquation[1] = coef * (double(pATA[0]) * double(ATV[0][1]) - double(pATA[2]) * double(ATV[0][0]));

        /*compute line fit error */
        for (int i = 0; i < minLineLen; i++) {
            coef = double(yCors[offset]) - double(xCors[offset++]) * lineEquation[0] - lineEquation[1];
            fitError += coef * coef;
        }
        return sqrt(fitError);
    }
    /*If the first pixel in this chain is vertical,
	 *then we try to find a vertical line, x=ay+b;*/
    if (pdirImg[yCors[offsetS] * imageWidth + xCors[offsetS]] == Vertical) {
        /*Build the system,and solve it using least square regression: mat * [a,b]^T = vec
		 * [y0,1]         [x0]
		 * [y1,1] [a]     [x1]
		 *    .   [b]  =   .
		 * [yn,1]         [xn]*/
        pMatT = fitMatT.ptr<float>();  //fitMatT = [y0, y1, ... yn; 1,1,...,1];
        for (int i = 0; i < minLineLen; i++) {
            //*(pMatT+minLineLen_) = 1;//the value are not changed;
            *(pMatT++) = yCors[offsetS];
            fitVec[0][i] = xCors[offsetS++];
        }
        ATA = fitMatT * (fitMatT.t());
        ATV = fitMatT * fitVec.t();
        /* [a,b]^T = Inv(mat^T * mat) * mat^T * vec */
        pATA = ATA.ptr<float>();
        coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
        //		lineEquation = svd.Invert(ATA) * matT * vec;
        lineEquation[0] = coef * (double(pATA[3]) * double(ATV[0][0]) - double(pATA[1]) * double(ATV[0][1]));
        lineEquation[1] = coef * (double(pATA[0]) * double(ATV[0][1]) - double(pATA[2]) * double(ATV[0][0]));
        /*compute line fit error */
        for (int i = 0; i < minLineLen; i++) {
            coef = double(xCors[offset]) - double(yCors[offset++]) * lineEquation[0] - lineEquation[1];
            fitError += coef * coef;
        }
        return sqrt(fitError);
    }
    return 0;
}

double EDLineDetector::EDLineDetectorParallel::LeastSquaresLineFit(cv::Mat_<float> &ATA, cv::Mat_<float> &ATV, cv::Mat_<float> &tempMatLineFit, cv::Mat_<float> &tempVecLineFit,
                                                                   unsigned int *xCors, unsigned int *yCors,
                                                                   unsigned int offsetS, unsigned int newOffsetS,
                                                                   unsigned int offsetE, std::array<double, 2> &lineEquation) const {
    int length = offsetE - offsetS;
    int newLength = offsetE - newOffsetS;
    if (length <= 0 || newLength <= 0) {
        cout << "EDLineDetector::LeastSquaresLineFit_ Error:"
                " the expected line index is wrong...offsetE = "
             << offsetE << ", offsetS=" << offsetS << ", newOffsetS=" << newOffsetS << endl;
        return -1;
    }
    if (lineEquation.size() != 2) {
        std::cout << "SHOULD NOT BE != 2" << std::endl;
        CV_Assert(false);
    }
    cv::Mat_<float> matT(2, newLength);
    cv::Mat_<float> vec(newLength, 1);
    float *pMatT;
    float *pATA;
    //	double fitError = 0;
    double coef;
    unsigned char *pdirImg = dirImg.data;
    /*If the first pixel in this chain is horizontal,
	 *then we try to find a horizontal line, y=ax+b;*/
    if (pdirImg[yCors[offsetS] * imageWidth + xCors[offsetS]] == Horizontal) {
        /*Build the new system,and solve it using least square regression: mat * [a,b]^T = vec
		 * [x0',1]         [y0']
		 * [x1',1] [a]     [y1']
		 *    .    [b]  =   .
		 * [xn',1]         [yn']*/
        pMatT = matT.ptr<float>();  //matT = [x0', x1', ... xn'; 1,1,...,1]
        for (int i = 0; i < newLength; i++) {
            *(pMatT + newLength) = 1;
            *(pMatT++) = xCors[newOffsetS];
            vec[0][i] = yCors[newOffsetS++];
        }
        /* [a,b]^T = Inv(ATA + mat^T * mat) * (ATV + mat^T * vec) */
        tempMatLineFit = matT * matT.t();
        tempVecLineFit = matT * vec;
        ATA = ATA + tempMatLineFit;
        ATV = ATV + tempVecLineFit;
        pATA = ATA.ptr<float>();
        coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
        lineEquation[0] = coef * (double(pATA[3]) * double(ATV[0][0]) - double(pATA[1]) * double(ATV[0][1]));
        lineEquation[1] = coef * (double(pATA[0]) * double(ATV[0][1]) - double(pATA[2]) * double(ATV[0][0]));
        /*compute line fit error */
        //		for(int i=0; i<length; i++){
        //			coef = double(yCors[offsetS]) - double(xCors[offsetS++]) * lineEquation[0] - lineEquation[1];
        //			fitError += coef*coef;
        //		}
        return 0;
    }
    /*If the first pixel in this chain is vertical,
	 *then we try to find a vertical line, x=ay+b;*/
    if (pdirImg[yCors[offsetS] * imageWidth + xCors[offsetS]] == Vertical) {
        /*Build the system,and solve it using least square regression: mat * [a,b]^T = vec
		 * [y0',1]         [x0']
		 * [y1',1] [a]     [x1']
		 *    .    [b]  =   .
		 * [yn',1]         [xn']*/
        //		pMatT = matT.GetData();//matT = [y0', y1', ... yn'; 1,1,...,1]
        pMatT = matT.ptr<float>();  //matT = [y0', y1', ... yn'; 1,1,...,1]
        for (int i = 0; i < newLength; i++) {
            *(pMatT + newLength) = 1;
            *(pMatT++) = yCors[newOffsetS];
            vec[0][i] = xCors[newOffsetS++];
        }
        /* [a,b]^T = Inv(ATA + mat^T * mat) * (ATV + mat^T * vec) */
        //		matT.MultiplyWithTransposeOf(matT, tempMatLineFit);
        tempMatLineFit = matT * matT.t();
        tempVecLineFit = matT * vec;
        ATA = ATA + tempMatLineFit;
        ATV = ATV + tempVecLineFit;
        //		pATA = ATA.GetData();
        pATA = ATA.ptr<float>();
        coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
        lineEquation[0] = coef * (double(pATA[3]) * double(ATV[0][0]) - double(pATA[1]) * double(ATV[0][1]));
        lineEquation[1] = coef * (double(pATA[0]) * double(ATV[0][1]) - double(pATA[2]) * double(ATV[0][0]));
        /*compute line fit error */
        //		for(int i=0; i<length; i++){
        //			coef = double(xCors[offsetS]) - double(yCors[offsetS++]) * lineEquation[0] - lineEquation[1];
        //			fitError += coef*coef;
        //		}
    }
    return 0;
}

bool EDLineDetector::EDLineDetectorParallel::LineValidation(unsigned int *xCors, unsigned int *yCors,
                                                            unsigned int offsetS, unsigned int offsetE,
                                                            std::array<double, 3> &lineEquation, float &direction) const {
    if (bValidate) {
        int n = offsetE - offsetS;
        /*first compute the direction of line, make sure that the dark side always be the
		 *left side of a line.*/
        int meanGradientX = 0, meanGradientY = 0;
        const short *pdxImg = dxImg.ptr<short>();
        const short *pdyImg = dyImg.ptr<short>();
        double dx, dy;
        std::vector<double> pointDirection;
        int index;
        for (int i = 0; i < n; i++) {
            index = yCors[offsetS] * imageWidth + xCors[offsetS++];
            meanGradientX += pdxImg[index];
            meanGradientY += pdyImg[index];
            dx = (double)pdxImg[index];
            dy = (double)pdyImg[index];
            pointDirection.push_back(atan2(-dx, dy));
        }
        dx = fabs(lineEquation[1]);
        dy = fabs(lineEquation[0]);
        if (meanGradientX == 0 && meanGradientY == 0) {  //not possible, if happens, it must be a wrong line,
            return false;
        }
        if (meanGradientX > 0 && meanGradientY >= 0) {  //first quadrant, and positive direction of X axis.
            direction = atan2(-dy, dx);                 //line direction is in fourth quadrant
        }
        if (meanGradientX <= 0 && meanGradientY > 0) {  //second quadrant, and positive direction of Y axis.
            direction = atan2(dy, dx);                  //line direction is in first quadrant
        }
        if (meanGradientX < 0 && meanGradientY <= 0) {  //third quadrant, and negative direction of X axis.
            direction = atan2(dy, -dx);                 //line direction is in second quadrant
        }
        if (meanGradientX >= 0 && meanGradientY < 0) {  //fourth quadrant, and negative direction of Y axis.
            direction = atan2(-dy, -dx);                //line direction is in third quadrant
        }
        /*then check whether the line is on the border of the image. We don't keep the border line.*/
        if (fabs(direction) < 0.15 || M_PI - fabs(direction) < 0.15) {                           //Horizontal line
            if (fabs(lineEquation[2]) < 10 || fabs(imageHeight - fabs(lineEquation[2])) < 10) {  //upper border or lower border
                return false;
            }
        }
        if (fabs(fabs(direction) - M_PI * 0.5) < 0.15) {                                        //Vertical line
            if (fabs(lineEquation[2]) < 10 || fabs(imageWidth - fabs(lineEquation[2])) < 10) {  //left border or right border
                return false;
            }
        }
        //count the aligned points on the line which have the same direction as the line.
        double disDirection;
        int k = 0;
        for (int i = 0; i < n; i++) {
            disDirection = fabs(direction - pointDirection[i]);
            if (fabs(2 * M_PI - disDirection) < 0.392699 || disDirection < 0.392699) {  //same direction, pi/8 = 0.392699081698724
                k++;
            }
        }
        //now compute NFA(Number of False Alarms)
        double ret = nfa(n, k, 0.125, logNT);

        return (ret > 0);  //0 corresponds to 1 mean false alarm
    } else {
        return true;
    }
}

void EDLineDetector::EDLineDetectorParallel::operator()(const cv::Range &range) const {
    cv::Mat_<float> ATA = cv::Mat_<int>(2, 2);               //the previous matrix of A^T * A;
    cv::Mat_<float> ATV = cv::Mat_<int>(1, 2);               //the previous vector of A^T * V;
    cv::Mat_<float> fitMatT = cv::Mat_<int>(2, minLineLen);  //the matrix used in line fit function;
    cv::Mat_<float> fitVec = cv::Mat_<int>(1, minLineLen);   //the vector used in line fit function;
    cv::Mat_<float> tempMatLineFit = cv::Mat_<int>(2, 2);    //the matrix used in line fit function;
    cv::Mat_<float> tempVecLineFit = cv::Mat_<int>(1, 2);    //the vector used in line fit function;

    for (int i = 0; i < minLineLen; i++) {
        fitMatT[1][i] = 1;
    }

    double lineFitErr = 0;  //the line fit error;
    unsigned int newOffsetS = 0;
    std::array<double, 2> lineEquation;
    unsigned int offsetInEdgeArrayS, offsetInEdgeArrayE, offsetInEdgeArrayS_init;  //start index and end index
    float direction;

    unsigned int *pEdgeXCors = &edges->xCors.front();
    unsigned int *pEdgeYCors = &edges->yCors.front();
    unsigned int *pEdgeSID = &edges->sId.front();
    unsigned char *pdirImg = dirImg.data;  //line direction

    for (int edgeID = range.start; edgeID < range.end; edgeID++) {
        offsetInEdgeArrayS = pEdgeSID[edgeID];
        offsetInEdgeArrayE = pEdgeSID[edgeID + 1];
        offsetInEdgeArrayS_init = offsetInEdgeArrayS;
        while (offsetInEdgeArrayE > offsetInEdgeArrayS + minLineLen) {  //extract line segments from an edge, may find more than one segments
            //find an initial line segment
            while (offsetInEdgeArrayE > offsetInEdgeArrayS + minLineLen) {
                lineFitErr = LeastSquaresLineFit(ATA, ATV, fitMatT, fitVec,
                                                 pEdgeXCors, pEdgeYCors, offsetInEdgeArrayS, lineEquation);
                if (lineFitErr <= lineFitErrThreshold)
                    break;                            //ok, an initial line segment detected
                offsetInEdgeArrayS += SkipEdgePoint;  //skip the first two pixel in the chain and try with the remaining pixels
            }
            if (lineFitErr > lineFitErrThreshold)
                break;  //no line is detected
            //An initial line segment is detected. Try to extend this line segment
            double coef1 = 0;       //for a line ax+by+c=0, coef1 = 1/sqrt(a^2+b^2);
            double pointToLineDis;  //for a line ax+by+c=0 and a point(xi, yi), pointToLineDis = coef1*|a*xi+b*yi+c|
            bool bExtended = true;
            bool bFirstTry = true;
            int numOfOutlier;  //to against noise, we accept a few outlier of a line.
            int tryTimes = 0;
            if (pdirImg[pEdgeYCors[offsetInEdgeArrayS] * imageWidth + pEdgeXCors[offsetInEdgeArrayS]] == Horizontal) {  //y=ax+b, i.e. ax-y+b=0
                                                                                                                        // if (false) {  //y=ax+b, i.e. ax-y+b=0
                offsetInEdgeArrayS_init = offsetInEdgeArrayS;
                while (bExtended) {
                    tryTimes++;
                    if (bFirstTry) {
                        bFirstTry = false;
                        offsetInEdgeArrayS += minLineLen;
                    } else {  //after each try, line is extended, line equation should be re-estimated
                        //adjust the line equation
                        lineFitErr = LeastSquaresLineFit(ATA, ATV, tempMatLineFit, tempVecLineFit,
                                                         pEdgeXCors, pEdgeYCors, offsetInEdgeArrayS_init, newOffsetS, offsetInEdgeArrayS, lineEquation);
                    }

                    coef1 = 1 / sqrt(lineEquation[0] * lineEquation[0] + 1);
                    numOfOutlier = 0;
                    newOffsetS = offsetInEdgeArrayS;
                    while (offsetInEdgeArrayE > offsetInEdgeArrayS) {
                        pointToLineDis = fabs(lineEquation[0] * pEdgeXCors[offsetInEdgeArrayS] - pEdgeYCors[offsetInEdgeArrayS] + lineEquation[1]) * coef1;
                        offsetInEdgeArrayS++;
                        if (pointToLineDis > lineFitErrThreshold) {
                            numOfOutlier++;
                            if (numOfOutlier > 3)
                                break;
                        } else {  //we count number of connective outliers.
                            numOfOutlier = 0;
                        }
                    }
                    //pop back the last few outliers from lines and return them to edge chain
                    offsetInEdgeArrayS -= numOfOutlier;
                    if (offsetInEdgeArrayS - newOffsetS > 0 && tryTimes < TryTime) {  //some new pixels are added to the line
                    } else {
                        bExtended = false;  //no new pixels are added.
                    }
                }
                //the line equation coefficients,for line w1x+w2y+w3 =0, we normalize it to make w1^2+w2^2 = 1.
                std::array<double, 3> lineEqu;
                lineEqu[0] = lineEquation[0] * coef1;
                lineEqu[1] = -1 * coef1;
                lineEqu[2] = lineEquation[1] * coef1;
                if (LineValidation(pEdgeXCors, pEdgeYCors, offsetInEdgeArrayS_init, offsetInEdgeArrayS, lineEqu, direction)) {
                    //store the line equation coefficients                                                                                                    //   lineEquations_.push_back( lineEqu );
                    Line line;
                    line.line_equation = lineEqu;
                    /*At last, compute the line endpoints and store them.
        *we project the first and last pixels in the pixelChain onto the best fit line
        *to get the line endpoints.
        *xp= (w2^2*x0-w1*w2*y0-w3*w1)/(w1^2+w2^2)
        *yp= (w1^2*y0-w1*w2*x0-w3*w2)/(w1^2+w2^2)  */
                    // std::array<float, 4> lineEndP;  //line endpoints

                    double a1 = lineEqu[1] * lineEqu[1];
                    double a2 = lineEqu[0] * lineEqu[0];
                    double a3 = lineEqu[0] * lineEqu[1];
                    double a4 = lineEqu[2] * lineEqu[0];
                    double a5 = lineEqu[2] * lineEqu[1];
                    unsigned int Px = pEdgeXCors[offsetInEdgeArrayS_init];  //first pixel
                    unsigned int Py = pEdgeYCors[offsetInEdgeArrayS_init];
                    float x1 = (float)(a1 * Px - a3 * Py - a4);  //x
                    float y1 = (float)(a2 * Py - a3 * Px - a5);  //y

                    Px = pEdgeXCors[offsetInEdgeArrayS - 1];  //last pixel
                    Py = pEdgeYCors[offsetInEdgeArrayS - 1];
                    float x2 = (float)(a1 * Px - a3 * Py - a4);  //x
                    float y2 = (float)(a2 * Py - a3 * Px - a5);  //y

                    line.line_endpoint[0] = x1;
                    line.line_endpoint[1] = y1;
                    line.line_endpoint[2] = x2;
                    line.line_endpoint[3] = y2;

                    line.center[0] = (x1 + x2) / 2.0;
                    line.center[1] = (y1 + y2) / 2.0;

                    line.length = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
                    {
                        lock.lock();
                        lines->push_back(line);
                        lock.unlock();
                    }
                }

            } else {  //x=ay+b, i.e. x-ay-b=0
                // offsetInLineArray = offsetInEdgeArrayS;
                offsetInEdgeArrayS_init = offsetInEdgeArrayS;
                while (bExtended) {
                    tryTimes++;
                    if (bFirstTry) {
                        bFirstTry = false;
                        offsetInEdgeArrayS += minLineLen;
                    } else {  //after each try, line is extended, line equation should be re-estimated
                              //adjust the line equation
                        lineFitErr = LeastSquaresLineFit(ATA, ATV, tempMatLineFit, tempVecLineFit,
                                                         pEdgeXCors, pEdgeYCors, offsetInEdgeArrayS_init, newOffsetS, offsetInEdgeArrayS, lineEquation);
                    }
                    coef1 = 1 / sqrt(1 + lineEquation[0] * lineEquation[0]);
                    numOfOutlier = 0;
                    newOffsetS = offsetInEdgeArrayS;
                    while (offsetInEdgeArrayE > offsetInEdgeArrayS) {
                        pointToLineDis = fabs(pEdgeXCors[offsetInEdgeArrayS] - lineEquation[0] * pEdgeYCors[offsetInEdgeArrayS] - lineEquation[1]) * coef1;
                        offsetInEdgeArrayS++;
                        if (pointToLineDis > lineFitErrThreshold) {
                            numOfOutlier++;
                            if (numOfOutlier > 3)
                                break;
                        } else {  //we count number of connective outliers.
                            numOfOutlier = 0;
                        }
                    }
                    //pop back the last few outliers from lines and return them to edge chain
                    offsetInEdgeArrayS -= numOfOutlier;
                    if (offsetInEdgeArrayS - newOffsetS > 0 && tryTimes < TryTime) {  //some new pixels are added to the line
                    } else {
                        bExtended = false;  //no new pixels are added.
                    }
                }
                //the line equation coefficients,for line w1x+w2y+w3 =0, we normalize it to make w1^2+w2^2 = 1.
                std::array<double, 3> lineEqu;
                lineEqu[0] = 1 * coef1;
                lineEqu[1] = -lineEquation[0] * coef1;
                lineEqu[2] = -lineEquation[1] * coef1;

                if (LineValidation(pEdgeXCors, pEdgeYCors, offsetInEdgeArrayS_init, offsetInEdgeArrayS, lineEqu, direction)) {
                    // if (LineValidation_(pLineXCors, pLineYCors, lineSID, offsetInLineArray, lineEqu, direction)) {  //check the line
                    //store the line equation coefficients

                    Line line;
                    line.line_equation = lineEqu;

                    /*At last, compute the line endpoints and store them.
                *we project the first and last pixels in the pixelChain onto the best fit line
                *to get the line endpoints.
                *xp= (w2^2*x0-w1*w2*y0-w3*w1)/(w1^2+w2^2)
                *yp= (w1^2*y0-w1*w2*x0-w3*w2)/(w1^2+w2^2)  */
                    // std::array<float, 4> lineEndP;  //line endpoints
                    double a1 = lineEqu[1] * lineEqu[1];
                    double a2 = lineEqu[0] * lineEqu[0];
                    double a3 = lineEqu[0] * lineEqu[1];
                    double a4 = lineEqu[2] * lineEqu[0];
                    double a5 = lineEqu[2] * lineEqu[1];
                    unsigned int Px = pEdgeXCors[offsetInEdgeArrayS_init];  //first pixel
                    unsigned int Py = pEdgeYCors[offsetInEdgeArrayS_init];
                    float x1 = (float)(a1 * Px - a3 * Py - a4);  //x
                    float y1 = (float)(a2 * Py - a3 * Px - a5);  //y
                    Px = pEdgeXCors[offsetInEdgeArrayS - 1];     //last pixel
                    Py = pEdgeYCors[offsetInEdgeArrayS - 1];
                    float x2 = (float)(a1 * Px - a3 * Py - a4);  //x
                    float y2 = (float)(a2 * Py - a3 * Px - a5);  //y

                    line.line_endpoint[0] = x1;
                    line.line_endpoint[1] = y1;
                    line.line_endpoint[2] = x2;
                    line.line_endpoint[3] = y2;

                    line.center[0] = (x1 + x2) / 2.0;
                    line.center[1] = (y1 + y2) / 2.0;

                    line.length = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));

                    {
                        lock.lock();
                        lines->push_back(line);
                        lock.unlock();
                    }
                }
            }
            //Extract line segments from the remaining pixel; Current chain has been shortened already.
        }
    }  //end for(unsigned int edgeID=0; edgeID<edges.numOfEdges; edgeID++)
}

int EDLineDetector::EDline(cv::Mat &image,
                           std::vector<Line> &lines,
                           bool smoothed) {
    //first, call EdgeDrawing function to extract edges
    if (!EdgeDrawing(image, edges_, smoothed)) {
        cout << "Line Detection not finished" << endl;
        return -1;
    }

    //detect lines
    logNT_ = 2.0 * (log10((double)imageWidth) + log10((double)imageHeight));

    lines.clear();
    lines.reserve(edges_.numOfEdges);

    cv::Mutex lock;

    EDLineDetectorParallel line_detector_parallel(this, &edges_, logNT_, lock, &lines);

    cv::parallel_for_(cv::Range(0, edges_.numOfEdges), line_detector_parallel, int(edges_.numOfEdges / 20));

    return 1;
}