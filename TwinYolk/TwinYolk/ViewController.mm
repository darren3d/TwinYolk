//
//  ViewController.m
//  TwinYolk
//
//  Created by darrenyao on 2017/3/18.
//  Copyright © 2017年 tencent. All rights reserved.
//

#import "ViewController.h"
#import "UIImage+CVMat.h"
#import <opencv2/opencv.hpp>
#import <opencv2/photo/photo.hpp>
#import <opencv2/nonfree/nonfree.hpp>
#import <opencv2/stitching/detail/seam_finders.hpp>
#import <opencv2/stitching/detail/blenders.hpp>
#import <opencv2/stitching/detail/exposure_compensate.hpp>
#import "seamless_cloning.hpp"

using namespace cv;

#include "graph.h"
typedef Graph<int,int,int> GraphType;

typedef struct ElementRGBA {
    uchar r, g, b, a;
}ElementRGBA;

typedef struct ElementRGB {
    uchar r, g, b;
}ElementRGB;

@interface ViewController ()
@property (nonatomic, weak) IBOutlet UIImageView *imageViewL;
@property (nonatomic, weak) IBOutlet UIImageView *imageViewR;
@property (nonatomic, weak) IBOutlet UIImageView *imageViewBlend;


@property (nonatomic, strong) UIImage *imageL;
@property (nonatomic, strong) UIImage *imageR;
@property (nonatomic, strong) UIImage *imageBlend;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    NSString *imagePathL = [[NSBundle mainBundle] pathForResource:@"test_1.jpg"
                                                           ofType:nil];
    NSString *imagePathR = [[NSBundle mainBundle] pathForResource:@"test_2.jpg"
                                                           ofType:nil];
    
    _imageL = [UIImage imageWithContentsOfFile:imagePathL];
    _imageR = [UIImage imageWithContentsOfFile:imagePathR];
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [self testImageBlend];
        dispatch_async(dispatch_get_main_queue(), ^{
            self.imageViewL.image = self.imageL;
            self.imageViewR.image = self.imageR;
            self.imageViewBlend.image = self.imageBlend;
        });
    });
}

- (void)testImageBlend {
    cv::Mat matBlend;
    cv::Mat matL = _imageL.getCVMat;
    cv::Mat matR = _imageR.getCVMat;
    cv::cvtColor(matL, matL, CV_RGBA2RGB);
    cv::cvtColor(matR, matR, CV_RGBA2RGB);
    
    cv::Mat grayL, grayR;
    cv::cvtColor(matL, grayL, CV_RGBA2GRAY);
    cv::cvtColor(matR, grayR, CV_RGBA2GRAY);
    
    std::vector<cv::Mat> imgs;
    imgs.push_back(matL);
    imgs.push_back(matR);
    
    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);
    vector<KeyPoint> keyPointsL, keyPointsR;
    
    detector.detect(grayL, keyPointsL);
    detector.detect(grayR, keyPointsR);
    
    SurfDescriptorExtractor descripExtractor;
    cv::Mat descripsL, descripsR;
    descripExtractor.compute(grayL, keyPointsL, descripsL);
    descripExtractor.compute(grayR, keyPointsR, descripsR);
    
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descripsL, descripsR, matches);
    size_t matchCount = matches.size();
    sort(matches.begin(),matches.begin()+matchCount);
    
    vector<Point2f> pointsL;
    vector<Point2f> pointsR;
    
    for(int i=0; i<matchCount; i++) {
        pointsL.push_back(keyPointsL[matches[i].queryIdx].pt);
        pointsR.push_back(keyPointsR[matches[i].trainIdx].pt);
    }
    
    cv::Mat homo = cv::findHomography(pointsL, pointsR, RANSAC);
    cv::Mat shftMat = (cv::Mat_<double>(3,3)<<1.0,0,grayL.cols, 0,1.0,0, 0,0,1.0);
    
    
    
    //拼接图像
    cv::Mat matRectifyL, matTitled(matR.rows, matL.cols + matR.cols, matR.type());
    cv::warpPerspective(matL, matRectifyL, homo, cv::Size(matL.cols, matL.rows));
    
    //seam find
//    cv::Mat seamL(matRectifyL.rows, matRectifyL.cols, CV_8UC1), seamR(matR.rows, matR.cols, CV_8UC1);
//    {
//        vector<cv::Mat> images;
//        vector<cv::Mat> seams;
//        images.push_back(matL);
//        images.push_back(matR);
//        seams.push_back(seamL);
//        seams.push_back(seamR);
//        
//        vector<cv::Point> corners;
//        corners.push_back(cv::Point(0, 0));
//        corners.push_back(cv::Point(0, 0));
//        
//        cv::Ptr<detail::SeamFinder> seamFinder = cv::makePtr<detail::GraphCutSeamFinder>();
//        seamFinder->find(images, corners, seams);
//        
//        seamL = seams[0];
//        seamR = seams[1];
//    }

    
    //mask 计算
    cv::Mat maskL(matRectifyL.rows, matRectifyL.cols, CV_8UC1);
    {
        maskL.setTo(cv::Scalar::all(0));
        cv::Mat temp(matRectifyL.rows, matRectifyL.cols, CV_8UC1);
        temp.setTo(cv::Scalar::all(255));
        cv::warpPerspective(temp, maskL, homo, cv::Size(matRectifyL.cols, matRectifyL.rows), INTER_NEAREST);
    }
    
    [self testSeamCutWithMatL:matRectifyL
                         matR:matR
                        maskL:maskL];
    return;

//    matRectifyL.mul(maskL);
//    matRectifyL = 0.5*matRectifyL + 0.5*matR;
    
    cv::Mat edgeL;
    cv::Sobel(matL, edgeL, matL.channels(), 1, 1);
    if(edgeL.channels() == 1){
        vector<cv::Mat>temp;
        temp.push_back(edgeL);
        temp.push_back(edgeL);
        temp.push_back(edgeL);
        cv::merge(temp, edgeL);
    }
    
    cv::Mat edgeR;
    cv::Sobel(matR, edgeR, matR.channels(), 1, 1);
    if(edgeR.channels() == 1){
        vector<cv::Mat>temp;
        temp.push_back(edgeR);
        temp.push_back(edgeR);
        temp.push_back(edgeR);
        cv::merge(temp, edgeR);
    }
    
    cv::Mat maskEdge;
    cv::Sobel(maskL, maskEdge, maskL.channels(), 1, 1);
    if(maskEdge.channels() == 1){
        vector<cv::Mat>temp;
        temp.push_back(maskEdge);
        temp.push_back(maskEdge);
        temp.push_back(maskEdge);
        cv::merge(temp, maskEdge);
    }

    
    edgeL.copyTo(Mat(matTitled, cv::Rect(0, 0, matL.cols, matL.rows)));
    maskEdge.copyTo(Mat(matTitled, cv::Rect(0, 0, matL.cols, matL.rows)));
    edgeR.copyTo(Mat(matTitled, cv::Rect(matL.cols, 0, matR.cols, matR.rows)));
    
    self.imageBlend = [UIImage initWithCVMat:matTitled];
    
    
    
//    seamlessClone(<#InputArray src#>, <#InputArray dst#>, <#InputArray mask#>, <#Point p#>, <#OutputArray blend#>, <#int flags#>)
}

- (cv::Rect)getMaxCommonRectFromMask:(const cv::Mat &)mask {
    int minX = mask.cols-1, maxX = 0, minY = mask.rows - 1, maxY = 0;
    for (int i=0; i<mask.rows; i++) {
        const uchar *maskData = mask.ptr<uchar>(i);
        for (int j=0; j<mask.cols; j++) {
            if (maskData[j] > 65) {
                minX = MIN(minX, j);
                maxX = MAX(maxX, j);
                
                minY = MIN(minY, i);
                maxY = MAX(maxY, i);
            }
        }
    }
    
    cv::Rect rect(minX, minY, maxX-minX+1, maxY-minY+1);
    return rect;
}

- (void)testSeamCutWithMatL:(cv::Mat &)matL
                       matR:(cv::Mat &)matR
                       maskL:(cv::Mat)maskL  {
    //填补mat的空白区域
    {
        cv::Mat temp = matR.clone();
        matL.copyTo(temp, maskL);
        matL = temp;
    }
    
    //self.imageBlend = [UIImage initWithCVMat:matR];
    cv::Mat maskR = 255 - maskL;
    //曝光补偿
    float rgbDiff[3] = {0, 0, 0};
    {
        int count = 0;
        for (int i = 0; i < matL.rows; i++) {
            uchar *eleMask = maskL.ptr<uchar>(i);
            ElementRGB *eleL = matL.ptr<ElementRGB>(i);
            ElementRGB *eleR = matR.ptr<ElementRGB>(i);
            for (int j = 0; j < matL.cols; j++) {
                if (eleMask[j] == 255) {
                    rgbDiff[0] += eleR[j].r - eleL[j].r;
                    rgbDiff[1] += eleR[j].g - eleL[j].g;
                    rgbDiff[2] += eleR[j].b - eleL[j].b;
                }
                count++;
            }
        }
        if (count > 0) {
            rgbDiff[0] /= count*2;
            rgbDiff[1] /= count*2;
            rgbDiff[2] /= count*2;
            
            //adjust rgb
            for (int i = 0; i < matL.rows; i++) {
                ElementRGB *eleL = matL.ptr<ElementRGB>(i);
                ElementRGB *eleR = matR.ptr<ElementRGB>(i);
                for (int j = 0; j < matL.cols; j++) {
                    int temp = eleR[j].r - rgbDiff[0];
                    temp = MAX(temp, 0);
                    eleR[j].r = temp;
                    temp = eleR[j].g - rgbDiff[1];
                    temp = MAX(temp, 0);
                    eleR[j].g = temp;
                    temp = eleR[j].b - rgbDiff[2];
                    temp = MAX(temp, 0);
                    eleR[j].b = temp;
                    
                    temp = eleL[j].r + rgbDiff[0];
                    temp = MIN(temp, 255);
                    eleL[j].r = temp;
                    temp = eleL[j].g + rgbDiff[1];
                    temp = MIN(temp, 255);
                    eleL[j].g = temp;
                    temp = eleL[j].b + rgbDiff[2];
                    temp = MIN(temp, 255);
                    eleL[j].b = temp;
                }
            }
        }
        
    
//        cv::Ptr<detail::ExposureCompensator> compensator = detail::ExposureCompensator::createDefault(detail::ExposureCompensator::GAIN);
//        std::vector<cv::Point> corners;
//        corners.push_back(cv::Point(0, 0));
//        corners.push_back(cv::Point(0, 0));
//        
//        std::vector<cv::Mat> images;
//        images.push_back(matL);
//        images.push_back(matR);
//        
//        //std::vector<std::pair<Mat,uchar> > masks;
//        //masks.push_back(std::make_pair(maskL, 0));
//        //masks.push_back(std::make_pair(maskR, 1));
//        std::vector<cv::Mat> masks;
//        masks.push_back(maskL);
//        masks.push_back(maskR);
//        compensator->feed(corners,
//                          images,
//                          masks);
//        compensator->apply(0, cv::Point(0, 0), matL, maskL);
//        compensator->apply(1, cv::Point(0, 0), matR, maskR);
    }
    
    cv::Mat matDiff(matL.rows, matL.cols, CV_8UC1);
    {
        matDiff.setTo(cv::Scalar::all(0));
        cv::Mat matBlurL, matBlurR;
        cv::adaptiveBilateralFilter(matL, matBlurL, cv::Size(11, 11), 6, 20);
        cv::adaptiveBilateralFilter(matR, matBlurR, cv::Size(11, 11), 6, 20);
        
        for (int i = 0; i < matL.rows; i++) {
            uchar *eleDiff = matDiff.ptr<uchar>(i);
            uchar *eleMask = maskL.ptr<uchar>(i);
            ElementRGB *eleL = matBlurL.ptr<ElementRGB>(i);
            ElementRGB *eleR = matBlurR.ptr<ElementRGB>(i);
            for (int j = 0; j < matL.cols; j++) {
                //                eleMask[j].a = 255;
                if (eleMask[j] == 255) {
                    float diff = (eleL[j].r - eleR[j].r) * (float)(eleL[j].r - eleR[j].r);
                    diff += (eleL[j].g - eleR[j].g) * (float)(eleL[j].g - eleR[j].g);
                    diff += (eleL[j].b - eleR[j].b) * (float)(eleL[j].b - eleR[j].b);
                    diff = sqrtf(diff*0.33333f);
                    //                    diff = (diff - 1)/diff;
                    eleDiff[j] = floorf(diff);
                }
            }
        }
        
        //        matR = matBlurR;
        
        cv::Mat matDiffErode;
        Mat kernel(5,5,CV_8U,Scalar(1));
        cv::erode(matDiff, matDiffErode, kernel, cv::Point(-1,-1), 2);
        cv::dilate(matDiffErode, matDiff, kernel, cv::Point(-1,-1), 2);
    }
    
    cv::Rect rect = [self getMaxCommonRectFromMask:matDiff];
    
    int overlap_width = rect.size().width;
    int xoffset = rect.x;
    
    int est_nodes = matL.rows * overlap_width;
    int est_edges = est_nodes * 4;
    
    GraphType g(est_nodes, est_edges);
    
    for(int i=0; i < est_nodes; i++) {
        g.add_node();
    }
    
    // Set the source/sink weights
    for(int y=0; y < matL.rows; y++) {
        g.add_tweights(y*overlap_width + 0, INT_MAX, 0);
        g.add_tweights(y*overlap_width + overlap_width-1, 0, INT_MAX);
    }
    
    // Set edge weights
    for(int y=0; y < matL.rows; y++) {
        for(int x=0; x < overlap_width; x++) {
            int idx = y*overlap_width + x;
            
            Vec3b a0 = matL.at<Vec3b>(y, xoffset + x);
            Vec3b b0 = matR.at<Vec3b>(y, xoffset + x);
            double cap0 = norm(a0, b0);
            
            // Add right edge
            if(x+1 < overlap_width) {
                Vec3b a1 = matL.at<Vec3b>(y, xoffset + x + 1);
                Vec3b b1 = matR.at<Vec3b>(y, xoffset + x + 1);
                
                double cap1 = norm(a1, b1);
                
                g.add_edge(idx, idx + 1, (int)(cap0 + cap1), (int)(cap0 + cap1));
            }
            
            // Add bottom edge
            if(y+1 < matL.rows) {
                Vec3b a2 = matL.at<Vec3b>(y+1, xoffset + x);
                Vec3b b2 = matR.at<Vec3b>(y+1, xoffset + x);
                
                double cap2 = norm(a2, b2);
                
                g.add_edge(idx, idx + overlap_width, (int)(cap0 + cap2), (int)(cap0 + cap2));
            }
        }
    }
    
    int flow = g.maxflow();
    std::cout << "max flow: " << flow << std::endl;
    
    int idx = 0;
    for(int y=0; y < matL.rows; y++) {
        idx = y*overlap_width;
        int maxX = 0;
        for(int x=0; x < overlap_width; x++) {
            if(g.what_segment(idx) != GraphType::SOURCE) {
                maxX = xoffset + x;
                break;
            }
            idx++;
        }
        
        uchar *eleMask = maskL.ptr<uchar>(y);
        for (int x=maxX+1; x<maskL.cols; x++) {
            eleMask[x] = 0;
        }
    }
    
    //图像融合
    cv::Mat matBlend;
    {
        cv::boxFilter(maskL, maskL, -1, cv::Size(41,41));
        matBlend = matR.clone();
        DY::seamlessClone(matL, matR, maskL, cv::Point(matR.cols/2, matR.rows/2), matBlend, DY::NORMAL_CLONE);
        
//        cv::boxFilter(maskL, maskL, -1, cv::Size(41,41));
//        maskR = 255 - maskL;
        
//        {
//            detail::MultiBandBlender blender;
//            blender.setNumBands(3);
//            blender.prepare(cv::Rect(0, 0, matL.cols, matL.rows));
//            blender.feed(matL, maskL, cv::Point(0,0));
//            
//            
//            blender.feed(matR, maskR, cv::Point(0,0));
//            {
//                cv::Mat ignore;
//                blender.blend(matBlend, ignore);
//                matBlend.convertTo(matBlend, (matBlend.type() / 8) * 8);
//            }
//        }

        
        
//        cv::GaussianBlur(maskL, maskL, cv::Size(7,7), 3);
//        maskR = 255 - maskL;
//        {
//            std::vector<cv::Mat> maskLs;
//            maskLs.push_back(maskL);
//            maskLs.push_back(maskL);
//            maskLs.push_back(maskL);
//            std::vector<cv::Mat> maskRs;
//            maskRs.push_back(maskR);
//            maskRs.push_back(maskR);
//            maskRs.push_back(maskR);
//            matBlend = matL.mul(maskLs) + matR.mul(maskRs);
//        }
        
    }

    
    self.imageL = [UIImage initWithCVMat:matL];
    self.imageR = [UIImage initWithCVMat:matR];
    self.imageBlend = [UIImage initWithCVMat:matBlend];

}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
