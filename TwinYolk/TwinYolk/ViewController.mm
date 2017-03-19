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
#import <opencv2/nonfree/nonfree.hpp>
#import <opencv2/stitching/detail/seam_finders.hpp>

using namespace cv;

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
    cv::Mat maskL(matRectifyL.rows, matRectifyL.cols, CV_8UC3);
    {
        maskL.setTo(cv::Scalar::all(0));
        cv::Mat temp(matRectifyL.rows, matRectifyL.cols, CV_8UC3);
        temp.setTo(cv::Scalar::all(255));
        cv::warpPerspective(temp, maskL, homo, cv::Size(matRectifyL.cols, matRectifyL.rows), INTER_NEAREST);
    }
    {
        cv::Mat matBlurL, matBlurR;
        cv::adaptiveBilateralFilter(matRectifyL, matBlurL, cv::Size(11, 11), 6, 20);
        cv::adaptiveBilateralFilter(matR, matBlurR, cv::Size(11, 11), 6, 20);
        
        for (int i = 0; i < matRectifyL.rows; i++) {
            ElementRGB *eleMask = maskL.ptr<ElementRGB>(i);
            ElementRGB *eleL = matBlurL.ptr<ElementRGB>(i);
            ElementRGB *eleR = matBlurR.ptr<ElementRGB>(i);
            for (int j = 0; j < matRectifyL.cols; j++) {
//                eleMask[j].a = 255;
                if (eleMask[j].r == 255) {
                    float diff = (eleL[j].r - eleR[j].r) * (float)(eleL[j].r - eleR[j].r);
                    diff += (eleL[j].g - eleR[j].g) * (float)(eleL[j].g - eleR[j].g);
                    diff += (eleL[j].b - eleR[j].b) * (float)(eleL[j].b - eleR[j].b);
                    diff = sqrtf(diff*0.33333f);
//                    diff = (diff - 1)/diff;
//                    diff *= 255;
                    
                    eleMask[j].r = diff;
                    eleMask[j].g = diff;
                    eleMask[j].b = diff;
                }
            }
        }
        
        matR = matBlurR;
        
        cv::Mat maskLErode;
        Mat kernel(5,5,CV_8U,Scalar(1));
        cv::erode(maskL, maskLErode, kernel, cv::Point(-1,-1), 2);
        cv::dilate(maskLErode, maskL, kernel, cv::Point(-1,-1), 2);
    }
    
    
    

//    matRectifyL.mul(maskL);
    matRectifyL = 0.5*matRectifyL + 0.5*matR;
    
    matRectifyL.copyTo(Mat(matTitled, cv::Rect(0, 0, matL.cols, matL.rows)));
    maskL.copyTo(Mat(matTitled, cv::Rect(0, 0, matL.cols, matL.rows)));
    matR.copyTo(Mat(matTitled, cv::Rect(matL.cols, 0, matR.cols, matR.rows)));
    
    self.imageBlend = [UIImage initWithCVMat:matTitled];
    
    
    
//    seamlessClone(<#InputArray src#>, <#InputArray dst#>, <#InputArray mask#>, <#Point p#>, <#OutputArray blend#>, <#int flags#>)
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
